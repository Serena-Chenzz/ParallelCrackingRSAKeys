#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

module load CUDA/7.5.18-GCC-4.9.2
module load GMP/5.1.3-GCC-4.9.2
make
time ./main 256-keys.txt 256
