#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=47G
#SBATCH --time=0-05:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-cogdrive


echo "starting training"

module load python/3.6
module load scipy-stack
source env_setup.sh
python -u train_conditional.py -e 2

#nvidia-smi
