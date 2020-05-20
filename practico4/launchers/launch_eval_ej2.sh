#!/bin/bash
#SBATCH --job-name=gpgpu_practico3
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:01:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=renzo.gambone@fing.edu.uy
#SBATCH -o salida_cluster_l1_global.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source /etc/profile.d/modules.sh

cd ~/practico4

lscpu

nvidia-smi

make ej2

nvprof --print-gpu-trace ./ej2.x $1 $2 --benchmark -numdevices=1 # imagen, algoritmo
