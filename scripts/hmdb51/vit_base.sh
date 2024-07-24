python -m torch.distributed.launch --nproc_per_node 4 main.py \
--cfg configs/hmdb51/vit_base.yaml
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  sbatch /home/smuralid/scratch/Video-FocalNets/vit_scratch.sh 
else
  echo "The script finished before timing out."
  # Exit or perform any other necessary cleanup
fi
