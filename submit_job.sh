#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=47G
#SBATCH --time=0-04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-cogdrive


#./prep_data.sh -l

echo "starting training"

module load python/3.6
module load scipy-stack
source env_setup.sh
python train.py

#nvidia-smi