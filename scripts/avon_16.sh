#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=32:00:00

module restore felix-ml

srun python src/train.py experiment=avon_16
