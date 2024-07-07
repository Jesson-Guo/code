#!/bin/bash
python main.py \
    --cfg configs/smt/cifar100_large_224.yaml \
    --data-path /mnt/data/ztl/mycode/data \
    --pretrained weights/smt_large_22k.pth \
    --desc large
