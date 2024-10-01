for run_time in 1
    do
        for wd in 1e-2
        do
            python run.py --dataset mind --gpu_id 4  --project_name mind_deforec_normalized_type\
             --sample_num 70 --dropout_p 0.1 --wd $wd --min_lr 0.000001\
              --run_name sasrec_mind_sample_70_dropout_0.1_wd_${wd}_min_lr_0.000001_runtime${run_time} --run_time ${run_time};
        done
    done