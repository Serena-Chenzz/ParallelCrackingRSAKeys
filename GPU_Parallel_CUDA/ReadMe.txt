This is for GPU algorithm

No Slurm Method:
Step 1: switch to GPU mode
command:
*sinteractive --time=0:30:0 --partition=gpu --gres=gpu:1

Step 2: Load modules in Spartan
commands:
*module load CUDA/7.5.18-GCC-4.9.2
*module load GMP/5.1.3-GCC-4.9.2

Step 3: Use Makefile to compile 
command:
*make

Step 4: Input parameter
command: 
*mpiexec ./main keyfile keyNum
e.g. time mpiexec ./main 20K-keys.txt 20480

Slurm Method:
Please refer to the example slurm file "test_256.slurm".

Output: Cracked Keys.

Data Set:
256-keys.txt: 256 keys
2048-keys.txt: 2048 keys
4096-keys.txt: 4096 keys
20K-keys.txt: 20480 keys