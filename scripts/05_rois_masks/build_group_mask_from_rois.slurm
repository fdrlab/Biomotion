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


conda run -n glmsingle_env python -u build_group_mask_from_rois.py \
  --roi-dir /lustre/scratch/data/s.../GLMsingle/split_half/roi \
  --save-frac 0.20 \
  --out-prefix /lustre/scratch/data/.../group_masks_new/cortexROI_runIntersect \
  --ortho-out /lustre/scratch/data/.../group_masks_new/qc_cortexROI_cov20.png \
  --title "cortexROI run-intersect group mask (>=20%)" \
  --cut-coords 35 -20 25 \
  --resample-mask-to-ref

