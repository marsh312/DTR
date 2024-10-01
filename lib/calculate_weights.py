import torch


def fill_leaf_code_scores(
    effective_leaf_code_scores, complete_leaf_num, leaf_id_leaf_code_index_array, device
):
    # effective_leaf_code_scores: [Bs, N]
    Bs, _ = effective_leaf_code_scores.shape
    with torch.no_grad():
        compelete_leaf_scores = torch.full(
            (Bs, complete_leaf_num), -1e9, dtype=torch.float32, device=device
        )
        compelete_leaf_scores[
            torch.arange(Bs).unsqueeze(1), leaf_id_leaf_code_index_array
        ] = effective_leaf_code_scores
    return compelete_leaf_scores #[Bs, complete_leaf_num]


def get_max_child_scores_and_indexs(input_scores):
    # input_scores: [Bs, N]
    Bs, N = input_scores.shape
    input_scores = input_scores.view(Bs, N // 2, 2)
    max_scores, max_child_indicators = input_scores.max(dim=2)
    return max_scores, max_child_indicators #[Bs, N//2], [Bs, N//2]


@torch.no_grad()
def calculate_layer_expectation(complete_leaf_scores, ancestor_indexs_layer_wise, max_layer_id):
    # complete_leaf_scores: [Bs, Num_leafs]
    Bs, N = complete_leaf_scores.shape
    # layers: 1, 2, ..., max_layer_id-1
    # corresponding indexs: 0, 1, ..., max_layer_id-2
    approximate_expectations = torch.zeros(
        (Bs, max_layer_id - 1), dtype=torch.float32, device=complete_leaf_scores.device
    )
    max_child_indicator_matrix = torch.zeros(
        (Bs, max_layer_id - 1), dtype=torch.int64, device=complete_leaf_scores.device
    )
    cur_max_node_scores = complete_leaf_scores
    for i in range(max_layer_id - 1, 0, -1):
        # process the i-th layer, i = max_layer_id-1, ..., 2, 1
        cur_max_node_scores, cur_max_child_indicators = get_max_child_scores_and_indexs(
            cur_max_node_scores
        )  # [Bs, 2^(i+1)] --> [Bs, 2^i];
        # cur_max_node_scores is the max child's scores of the i-th layer, cur_max_child_indicators is the max child's indicator of the i-th layer
        approximate_expectations[:, i - 1] = cur_max_node_scores.sum(-1)
        max_child_indicator_matrix[:, i - 1] = cur_max_child_indicators[torch.arange(Bs), ancestor_indexs_layer_wise[:, i - 1]]

    return (
        approximate_expectations,
        max_child_indicator_matrix,
    )  # [Bs, max_layer_id-1], [Bs, max_layer_id-1]


@torch.no_grad()
def generate_ancestors_layer_wise(target_leaf_codes, max_tree_id, device):
    # target_leaf_codes: [Bs, 1]
    Bs, _ = target_leaf_codes.shape
    ancestor_codes = target_leaf_codes
    ancestor_indexs_layer_wise = torch.zeros(
        (Bs, max_tree_id - 1), dtype=torch.int64, device=device
    )
    for i in range(max_tree_id, 1, -1):
        target_leaf_codes = torch.div(target_leaf_codes - 1, 2, rounding_mode="floor")
        ancestor_codes = torch.cat((target_leaf_codes, ancestor_codes), dim=-1)
        ancestor_indexs_layer_wise[:, i - 2] = target_leaf_codes.view(-1) - (2**(i-1) - 1)
    assert (ancestor_codes >= 0).all() 
    # ancestor_codes[i] is all ancestors of target_leaf_codes[i] exculding root, j = 0, 1, ..., max_tree_id-1, corresponding to the j+1-th layer
    # ancestor_indexs_layer_wise[i, j] is the index of the j+1-th ancestor of target_leaf_codes[i] in the j+1-th layer, j = 0, 1, ..., max_tree_id-2
    return ancestor_codes, ancestor_indexs_layer_wise  # [Bs, max_tree_id], [Bs, max_tree_id-1]


@torch.no_grad()
def calulate_calibrated_weight_in_full_softmax(
    leaf_scores, target_leaf_codes, max_layer_id, leaf_id_leaf_code_index_array, device, calibrated_mode="normalized"
):
    # leaf_scores: [Bs, Num_leafs], target_leaf_codes: [Bs, 1]
    assert len(leaf_scores) == len(target_leaf_codes)
    Bs, N = leaf_scores.shape
    softmax_scores = torch.softmax(leaf_scores, dim=-1)
    complete_leaf_scores = fill_leaf_code_scores(
        softmax_scores, 2**max_layer_id, leaf_id_leaf_code_index_array, device
    )
    ancestor_codes, ancestor_indexs = generate_ancestors_layer_wise(
        target_leaf_codes, max_layer_id, device
    )
    # lefat_right_indicators[i,j] indicates the child in the j+2-th layer of ancestor_codes[i,j](j+1-th layer) is left or right
    # j = 0, 1, ..., max_layer_id-2
    left_right_indicators = ((ancestor_codes + 1) % 2)[:, 1:] 
    approximate_expectations, max_child_indicator_matirx = calculate_layer_expectation(
        complete_leaf_scores, ancestor_indexs, max_layer_id
    )
    unnormalized_weight = (left_right_indicators == max_child_indicator_matirx).float()
    if calibrated_mode == "normalized":
        calibrated_weight = unnormalized_weight / approximate_expectations
    elif calibrated_mode == "unnormalized":
        calibrated_weight = unnormalized_weight
    elif calibrated_mode == "tailored":
        calibrated_weight = unnormalized_weight / approximate_expectations
        maxs_revised = calibrated_weight.max(dim=-1, keepdim=True).values
        maxs_revised[maxs_revised == 0] = 1
        calibrated_weight = calibrated_weight / maxs_revised.expand_as(calibrated_weight.shape) 
    else:
        raise ValueError("calibrated_mode should be one of ['normalized', 'unnormalized', 'tailored']")
    # concatenate the weight of the last layer, weight is 1
    calibrated_weight = torch.cat(
        (calibrated_weight, torch.ones((Bs, 1), dtype=torch.float32, device=device)), dim=-1
    )
    return calibrated_weight.view(-1)

@torch.no_grad()
def calculate_adaptive_weight(batch_user, tree, network_model, all_leaf_codes, device, weight_with_grad=False):
    Bs, d = batch_user.shape
    adaptive_weight = torch.ones((Bs, tree.max_layer_id), device=device)
    node_codes = all_leaf_codes
    user_index = torch.arange(Bs, device=device).view(-1,1).repeat(1, 2)
    child_preference = torch.full((Bs, 2), -torch.inf, device=device)
    for layer in range(tree.max_layer_id, 1, -1):
        is_left = node_codes % 2 == 1
        sibling_codes = torch.where(is_left, node_codes + 1, node_codes - 1)
        codes = torch.cat((node_codes.view(-1,1), sibling_codes.view(-1,1)), dim=-1) #[Bs, 2]
        all_labels = tree.node_code_node_id_array[codes]
        effective_index = all_labels >= 0
        effective_labels = all_labels[effective_index]

        cur_user_index = user_index[effective_index].view(-1)
        new_batch_user = batch_user[cur_user_index]
        effective_item_index = new_batch_user >= 0
        new_batch_user[effective_item_index] = \
            tree.item_id_node_ancestor_id[new_batch_user[effective_item_index], layer]

        child_preference[:] = -torch.inf
        if weight_with_grad:
            child_preference[effective_index] = network_model.preference(new_batch_user, effective_labels.view(-1,1))[:, 0]
        else:
            with torch.no_grad():
                child_preference[effective_index] = network_model.preference(new_batch_user, effective_labels.view(-1,1))[:, 0]
        softmax_child_preference = torch.nn.functional.softmax(child_preference, dim=-1)
        adaptive_weight[:, layer-2] = softmax_child_preference[:, 0]
        node_codes = torch.div(node_codes-1, 2, rounding_mode='floor')
    return adaptive_weight.view(-1)

@torch.no_grad()
def calculate_max_weight(batch_user, tree, network_model, all_leaf_codes, device):
    Bs, d = batch_user.shape
    max_weight = torch.ones((Bs, tree.max_layer_id), device=device)
    node_codes = all_leaf_codes
    user_index = torch.arange(Bs, device=device).view(-1,1).repeat(1, 2)
    child_preference = torch.full((Bs, 2), -torch.inf, device=device)
    for layer in range(tree.max_layer_id, 1, -1): # layer = max_layer_id, max_layer_id-1, ..., 2
        is_left = node_codes % 2 == 1
        sibling_codes = torch.where(is_left, node_codes + 1, node_codes - 1)
        codes = torch.cat((node_codes.view(-1,1), sibling_codes.view(-1,1)), dim=-1) #[Bs, 2]
        all_labels = tree.node_code_node_id_array[codes]
        effective_index = all_labels >= 0
        effective_labels = all_labels[effective_index]

        cur_user_index = user_index[effective_index].view(-1)
        new_batch_user = batch_user[cur_user_index]
        effective_item_index = new_batch_user >= 0
        new_batch_user[effective_item_index] = \
            tree.item_id_node_ancestor_id[new_batch_user[effective_item_index], layer]

        child_preference[:] = -torch.inf
        with torch.no_grad():
            child_preference[effective_index] = network_model.preference(new_batch_user, effective_labels.view(-1,1))[:, 0]
        softmax_child_preference = torch.nn.functional.softmax(child_preference, dim=-1)
        max_weight[:, layer-2] = (softmax_child_preference[:,0] >= softmax_child_preference[:,1]).float() 
        node_codes = torch.div(node_codes-1, 2, rounding_mode='floor')
    return max_weight.view(-1) # [Bs * max_layer_id]

if __name__ == "__main__":
    from Tree_Model import Tree

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    codes = [15, 16, 18, 19, 22, 23, 25, 28, 29]
    tree = Tree(ids, codes)
    leaf_id_leaf_code_index_array = tree.leaf_index_complete_leaf_index_array
    complete_leaf_num = 2**tree.max_layer_id
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    effective_leaf_code_scores = torch.randint(1, 4, (3, 9), device=device).float()
    effective_leaf_code_scores[1][6] = 4
    complete_leaf_scores = fill_leaf_code_scores(
        effective_leaf_code_scores,
        complete_leaf_num,
        leaf_id_leaf_code_index_array,
        device,
    )
    print(complete_leaf_scores)
    print(complete_leaf_scores.shape)
    layer_num = tree.max_layer_id
    
    max_scores, max_indexs = get_max_child_scores_and_indexs(complete_leaf_scores)

    all_leaf_codes = torch.randint(15, 30, (3, 1), device=device).long()
    ancestor_codes, ancestor_indexs = generate_ancestors_layer_wise(all_leaf_codes, layer_num, device)
    approximate_expectations, max_indexs_matrix = calculate_layer_expectation(
        complete_leaf_scores, ancestor_indexs, layer_num,
    )
    calibrated_weight = calulate_calibrated_weight_in_full_softmax(
        effective_leaf_code_scores, all_leaf_codes, layer_num, leaf_id_leaf_code_index_array, device
    )
    print(approximate_expectations)
    print(approximate_expectations.shape)
