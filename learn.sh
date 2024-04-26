#!/bin/bash
#SBATCH -J PacLearn
#SBATCH --time=00-6:00:00
#SBATCH -p gpu,preempt
#SBATCH -N 1
#SBATCH	-n 32
#SBATCH --mem=16g
#SBATCH --gres=gpu:a100:1
#SBATCH --output=pacbot-learn.%j.out
#SBATCH --error=pacbot-learn.%j.err

module purge
hostname
module load anaconda/2021.05
module load cuda/12.2

module list
source activate cupy

python -m src

conda deactivate
