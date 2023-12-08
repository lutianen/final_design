#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final_design
DATA_DIR=/home/tianen/doc/MachineLearningData/
# Ubuntu Server
# PROJECT_PREFIX=/home/lutianen/final_design
# DATA_DIR=/home/tianen/data/

python ${PROJECT_PREFIX}/utils/get_flops.py \
    --arch vgg_cifar \
    --cfg vgg16 \
    --data_path ${DATA_DIR} \
    --job_dir ./experiment/cifar/vgg_1 \
    --pretrain_model ${PROJECT_PREFIX}/pretrain_model/vgg16_cifar10.pt \
    --lr 0.01 \
    --lr_decay_step 50 100 \
    --weight_decay 0.005  \
    --num_epochs 120 \
    --train_batch_size 512 \
    --gpus 0 \
    --pr_target 0.5 \
    --graph_gpu
