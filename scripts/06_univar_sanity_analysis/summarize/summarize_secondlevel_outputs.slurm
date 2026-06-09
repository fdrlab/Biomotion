#!/bin/bash
#SBATCH --job-name=tfce_qc
#SBATCH --output=tfce_qc_output_%j.txt
#SBATCH --error=tfce_qc_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_short
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

unset SLURM_EXPORT_ENV
set -euo pipefail

# Conda env
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


conda run -n glmsingle_env python -u /lustre/scratch/data/.../second_level/summarize_secondlevel_outputs.py \
    --scan-roots /lustre/scratch/data/.../second_level/GLMsingle_no_mask/ \
    --outdir /lustre/scratch/data/.../second_level/GLMsingle_no_mask_new/summary/