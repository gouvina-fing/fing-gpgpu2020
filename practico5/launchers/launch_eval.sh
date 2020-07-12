#!/bin/bash
#SBATCH --exclude=node[21,23-25]
#SBATCH --job-name=gpgpu_practico5
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -o salida_cluster_dtrsm_shared_4096.out

module load cuda/9.2

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source /etc/profile.d/modules.sh

cd ~/practico5

lscpu

nvidia-smi

# Timing

make cluster

nvprof --print-gpu-trace ./labgpu20.x $1 $2 $3 $4 --benchmark -numdevices=1 # algoritmo, tam1, tam2, tam3

# Memory and Shared Efficiency

#make cluster_no_l1

#nvprof --profile-api-trace none --metrics "gld_efficiency,gst_efficiency,shared_efficiency,achieved_occupancy" ./labgpu20_no_l1.x $1 $2 $3 $4 # algoritmo, tam1, tam2, tam3
