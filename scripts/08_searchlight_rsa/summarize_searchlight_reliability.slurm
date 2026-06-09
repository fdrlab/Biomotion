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


python summarize_searchlight_reliability.py \
  --root /lustre/scratch/data/.../searchlight_RSA \
  --outdir /lustre/scratch/data/.../searchlight_RSA/_QC_summary_unsmoothed_placebo_group \
  --approaches A_baseline B_up3x_cov20 C_up3x_cov20 \
  --datatypes unsmoothed \
  --metric crossnobis \
  --prefer-newest \
  --write-long-csv \
  --subjects sub-105 sub-106 sub-107 sub-112 sub-117 sub-118 sub-120 sub-122 sub-135 sub-136 sub-142 sub-144 sub-146 sub-147 sub-150 sub-151 sub-152 sub-153 sub-154 sub-155 sub-157 sub-164 sub-165 sub-171 sub-172 sub-174 sub-175 sub-176   # placebo participants