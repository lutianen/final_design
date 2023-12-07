#!/bin/bash

python /home/tianen/doc/_XiDian/___FinalDesign/FinalDesign_v0/train_test/train_cifar.py \
    --arch vgg_cifar \
    --cfg vgg16 \
    --data_path /home/tianen/doc/MachineLearningData/ \
    --job_dir ./experiment/cifar/vgg_1 \
    --pretrain_model //home/tianen/doc/_XiDian/___FinalDesign/FinalDesign_v0/pretrain_model/vgg16_cifar10.pt \
    --lr 0.01 \
    --lr_decay_step 50 100 \
    --weight_decay 0.005  \
    --num_epochs 120 \
    --train_batch_size 512 \
    --gpus 0 \
    --pr_target 0.5 \
    --graph_gpu
