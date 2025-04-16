#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=vf_test
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=96G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/video-focalnets/k600/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/video-focalnets/k600/slurm-%j.err

cd $SLURM_TMPDIR
cp -r /project/def-mpederso/smuralid/envs/focal_new.zip . 
unzip -qq focal_new.zip
module load StdEnv/2020 gcc/9.3.0 opencv/4.5.1 python/3.8.10
source focal/bin/activate
mkdir data && cd data
cp -r /project/def-mpederso/smuralid/datasets/kinetics600.zip .
cp /project/def-mpederso/smuralid/datasets/Daily-DA/ARID_v1_5_211015.zip .
unzip -qq kinetics600.zip
unzip -qq ARID_v1_5_211015.zip
mv clips_v1.5 arid
cd $SLURM_TMPDIR


cd Video-FocalNets 

timeout 170m source train.sh 
 
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  
  sbatch train_job.sh 
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi
