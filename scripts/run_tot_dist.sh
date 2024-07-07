#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 \
    main_tot.py \
    --cfg configs/tot/cifar100_large_224.yaml \
    --data-path /mnt/data/ztl/mycode/data \
    --pretrained weights/tot_large_22k.pth \
    --desc large \
    --tot-path configs/tots/cifar100-5.json \
    --dist
    # --loss ace \
