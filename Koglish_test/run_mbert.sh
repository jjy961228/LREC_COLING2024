#!/bin/bash
for seed in 11111 22222 33333 44444 55555
do
    for model in mbert_uncased 
    do
        for task in cola qnli sst2 stsb mrpc rte
        do
            for lang_type in en2en en2cross cross2cross
            do
                for schedular in linear
                do
                    for warmup_step in 0.1
                    do
                        for lr in 5e-5
                        do
                            CUDA_VISIBLE_DEVICES=0 python main.py --random_seed $seed --model $model --task $task --lang_type $lang_type --schedular $schedular --warmup_step $warmup_step --lr $lr
                        done
                    done
                done
            done
        done
    done
done


