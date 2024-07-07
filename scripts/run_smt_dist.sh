#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 \
    main.py \
    --cfg configs/smt/stanford_cars_large_448.yaml \
    --data-path /mnt/data/ztl/mycode/data \
    --pretrained weights/smt_large_22k.pth \
    --desc large \
    --dist
