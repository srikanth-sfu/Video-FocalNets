#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --account=def-mpederso
#SBATCH --mem-per-cpu=64G 
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:01
#SBATCH -o /home/smuralid/error/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/slurm-%j.err
#SBATCH --signal=B:USR1@30  # Send USR1 signal 30 seconds before time limit

# Directories

# Environment setup
source /home/smuralid/anaconda3/bin/activate
source activate focal

handle_timeout() {
  echo "The script timed out after the time limit. Restarting..."
  sbatch /home/smuralid/scratch/Video-FocalNets/vit_scratch_v2.sh 
  exit 0
}

trap 'handle_timeout' USR1

# Check if the timeout command's exit status is 124, which indicates a timeout occurred
python -m torch.distributed.launch --nproc_per_node 4 main.py \
--cfg configs/hmdb51/vit_base.yaml
echo $?
echo "Crossed path"
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  sbatch /home/smuralid/scratch/Video-FocalNets/vit_scratch_v2.sh 
else
  echo "The script finished before timing out."
  # Exit or perform any other necessary cleanup
fi