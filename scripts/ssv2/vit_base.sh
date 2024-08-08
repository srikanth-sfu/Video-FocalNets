python -m torch.distributed.launch --nnodes=4 --nproc-per-node=4 main.py \
--cfg configs/ssv2/vit_base.yaml
