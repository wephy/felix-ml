#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --partition=gpu
#SBATCH --acount=su007-rr-gpu

module restore felix-ml

srun python src/train.py