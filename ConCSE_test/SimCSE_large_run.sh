for model in xlmr_large
do
    for seed in 11111 22222 33333 44444 55555
    do
        for task in stsb sts12 sts13 sts14 sts15 sts16 sick
        do  
            for lang_type in en2en en2cross cross2cross
            do
                for schedular in linear
                do
                    for warmup_step in 0.1
                    do
                        for temp in 0.05   
                        do  
                            for lr in 1e-5
                            do
                                CUDA_VISIBLE_DEVICES=0 python main.py --model $model --random_seed $seed --eval_type transfer --method simcse --task $task --lang_type $lang_type --temp $temp --lr $lr
                            done
                        done
                    done
                done
            done
        done
    done
done