python -m torch.distributed.launch --nproc_per_node 2 main.py  --eval --cfg configs/ssv2/clip_base.yaml --resume 'test' --opts TEST.NUM_CLIP 1 TEST.NUM_CROP 3 DATA.ROOT path/to/root DATA.TRAIN_FILE data/ssv2/val.csv DATA.VAL_FILE data/ssv2/val.csv

