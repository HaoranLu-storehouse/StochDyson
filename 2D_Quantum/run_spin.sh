#!/bin/sh
#SBATCH -J Spin
#SBATCH -p q_ysuan
#SBATCH -o job_%j.log
#SBATCH -e job_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56

source activate base2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export POISSON_V1_WORKERS=48
export POISSON_V1_HI_WORKERS=48
export POISSON_V1_REPLICA_WORKERS=1
export POISSON_V1_FFT_WORKERS=8

python run_poisson_v1.py ./Spin_two_level/spin_two_level_reaction_coordinate.yaml

