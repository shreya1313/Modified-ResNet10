#!/bin/bash

#SBATCH --job-name=C5GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch_c2cpu
#SBATCH --output="main_output" 

module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;


singularity exec --nv \
	    --overlay /scratch/$USER/path/to/singularity.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python /path/to/gridsearch.py"
