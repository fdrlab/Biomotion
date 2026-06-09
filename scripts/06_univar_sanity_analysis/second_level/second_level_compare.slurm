#!/bin/bash
#SBATCH --job-name=tfce_qc
#SBATCH --output=tfce_qc_output_%j.txt
#SBATCH --error=tfce_qc_error_%j.txt
#SBATCH --time=02:00:00
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

#SUBS=(105 106 107 112 114 117 118 120 122 135 136 142 144 146 147 150 151 152 153 154 155 
#157 164 165 171 172 174 175 176)

conda run -n glmsingle_env python -u second_level_compare_new.py \
  --mask-img FULL \
  --outdir   /lustre/scratch/data/.../second_level/GLMsingle_no_mask/unsmoothed/B_up3x_cov30_no-mask_no_smoothing \
  --subjects sub-105 sub-106 sub-107 sub-112 sub-114 sub-117 sub-118 sub-120 sub-122 sub-135 sub-136 sub-142 sub-144 sub-146 sub-147 sub-150 sub-151 sub-152 sub-153 sub-154 sub-155 sub-157 sub-164 sub-165 sub-171 sub-172 sub-174 sub-175 sub-176 \
  --glms-dir    /lustre/scratch/data/.../nilearn_firstlevel/GLMsingle_new/B_up3x_no_smoothing \
  --runs-tag 01-02 \
  --datatype unsmoothed \
  --branch B_up3x \
  --unc-p 0.001 --unc-z 3.1 --unc-k 20 \
  --alpha 0.05 --cluster-k 10 \
  --one-sided \
  --perm-n 0 \
  --save-parametric-pmaps-pngs \
  --save-permutation-pngs \
  --design-order-json /lustre/scratch/data/.../GLMsingle/design_order_map.json
