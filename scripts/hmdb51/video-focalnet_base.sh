# python -m torch.distributed.launch --nproc_per_node 1 main.py --cfg configs/hmdb51/video-focalnet_base.yaml --resume 'ckpt/kinetics400/video-focalnet_base_kinetics400.pth' --opts DATA.TRAIN_FILE data/hmdb51/train.csv DATA.VAL_FILE data/hmdb51/val.csv
python -m torch.distributed.launch --nproc_per_node 1 main.py --eval --cfg configs/hmdb51/video-focalnet_base.yaml --resume 'ckpt/hmdb51/video_hmdb_51.pth' --opts  DATA.NUM_FRAMES 8 DATA.BATCH_SIZE 8 TEST.NUM_CLIP 4 TEST.NUM_CROP 3 DATA.TRAIN_FILE data/hmdb51/train.csv DATA.VAL_FILE data/hmdb51/val.csv 

