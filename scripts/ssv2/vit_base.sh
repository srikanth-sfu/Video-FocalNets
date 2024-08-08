#python -m torch.distributed.launch --nnodes=3 --nproc-per-node=4 main.py \
#--cfg configs/ssv2/vit_base.yaml
export NCCL_BLOCKING_WAIT=1
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 --master_port=29600 --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID main.py --cfg configs/ssv2/vit_base.yaml
