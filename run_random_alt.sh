#!/bin/bash
read -p "cuda: " cuda
read -p "sparse_init: " sparse_init
read -p "model : " model
read -p "data : " data
read -p "batch : " batch
read -p "epoch: " epoch
read -p "mask-pop: " pop
read -p "mask-sam: " sam
read -p "mask-iter: " miter
read -p "output-dir: " dir
read -p "skip generating [y, n]? : " skip


if [[ -z "$epoch" ]]; then
	epoch=250
fi

if [[ -z "$pop" ]]; then
	pop=3000
fi

if [[ -z "$sam" ]]; then
	sam=10
fi

if [ "$skip" == "n" ]; then

CUDA_VISIBLE_DEVICES=$cuda python lth_tickets_generation.py --sparse --seed 18 --multiplier 1 --lr \
0.1 --density 0.01 --update_frequency 1500 --epochs $epoch --model $model --data $data \
--decay_frequency 30000 --growth random --death magnitude --redistribution none --output-dir $dir \
--num-iteration $miter --mask-population $pop --mask-sampling $sam --sparse_init $sparse_init \
--batch $batch --alt

fi

CUDA_VISIBLE_DEVICES=$cuda python lth_tickets_training.py --sparse --seed 18 --multiplier 1 --lr 0.1 \
--density 0.01 --update_frequency 1500 --epochs $epoch --model $model --data $data --decay_frequency \
30000 --growth random --death magnitude --redistribution none --output-dir $dir --num-iteration \
$miter --mask-population $pop --mask-sampling $sam --sparse_init $sparse_init \
--skip-exist-full --batch-size $batch
