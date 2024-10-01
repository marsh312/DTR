import torch
from torch import nn
import math

def generate_padding_mask(inputs):
    padding_mask = torch.zeros(inputs.size())
    padding_mask[inputs == -1] = -torch.inf
    # return padding_mask.bool()
    return padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #[max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SASRec(nn.Module):
    def __init__(self, n_item, max_len, device, args):
        super(SASRec, self).__init__()
        self.n_item = n_item
        self.max_len = max_len  
        self.d_model = args.d_model
        self.device = device
        self.not_weights_as_linear = args.not_weights_as_linear

        # Define embedding layer
        self.item_embedding = nn.Embedding(n_item+1, self.d_model, padding_idx=-1)

        # Define position encoding layer
        if args.position_encoding == 'learned':
            self.pos_embedding = nn.Embedding(max_len, self.d_model)
        else:
            self.pos_encoder = PositionalEncoding(self.d_model)

        # Define encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=args.nhead,
                                                   dim_feedforward=args.dim_feedforward,
                                                   dropout=args.dropout,
                                                   batch_first=True)

        # Define layer-normalization layer
        self.norm = nn.LayerNorm(self.d_model)

        # Define transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers, norm=self.norm)

        # Define linear layer
        self.linear = nn.Linear(self.d_model, n_item)
    
    def forward(self, historical_items):
        # src_mask = generate_square_subsequent_mask(historical_items.size(1), self.device)
        src_key_padding_mask = generate_padding_mask(historical_items).to(self.device)
        
        historical_items[historical_items == -1] = self.n_item
        src = self.item_embedding(historical_items)
        assert not torch.isnan(src).any()
        if hasattr(self, 'pos_embedding'):
            src = src + self.pos_embedding(torch.arange(historical_items.size(1)).unsqueeze(0).to(self.device))
        else:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        assert not torch.isnan(output).any()
        output = output[:, -1, :]
        if self.not_weights_as_linear:
            output = self.linear(output)
        else:
            output = output @ self.item_embedding.weight.t()
        return output[:, :-1] #[bs, n_item]