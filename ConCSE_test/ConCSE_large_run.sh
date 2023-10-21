for model in xlmr_large
do
    for seed in 11111 22222 33333 44444 55555
    do
        for task in stsb sts12 sts13 sts14 sts15 sts16 sick
        do
            for ours_version in 999
            do  
                for schedular in linear
                do
                    for warmup_step in 0.1
                    do
                        for temp in 0.05   
                        do  
                            for margin in 1.0
                            do  
                                for lr in 1e-5
                                do
                                    for triplet in 1.2 
                                    do 
                                        CUDA_VISIBLE_DEVICES=0 python main.py --random_seed $seed --eval_type transfer --model $model --method ours --lang_type cross2cross --ours_version $ours_version --temp $temp --task $task --schedular $schedular --warmup_step $warmup_step --lr $lr --margin $margin --triplet $triplet
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done