#!/bin/bash
#SBATCH --job-name=searchlight
#SBATCH --output=searchlight_output_%A_%a.txt 
#SBATCH --error=searchlight_error_%A_%a.txt   
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2           
#SBATCH --mem=20G
#SBATCH --nodes=1                 
#SBATCH --partition=intelsr_short
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

set -euo pipefail
mkdir -p logs

# Conda env (same path you used before)
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Thread pinning
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


conda run -n glmsingle_env python -u collinearity_check.py \
  --root /lustre/scratch/data/.../searchlight_RSA \
  --datatype unsmoothed

