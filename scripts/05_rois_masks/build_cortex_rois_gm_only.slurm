#!/bin/bash
#SBATCH --job-name=tfce_qc
#SBATCH --output=tfce_qc_output_%j.txt
#SBATCH --error=tfce_qc_error_%j.txt
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_medium
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
mkdir -p logs

# If you used the filter step, use subjects_ready.txt; otherwise subjects_all.txt
SUBJECTS_FILE=/lustre/scratch/data/.../GLMsingle/subjects_all.txt
SUBJECTS=$(tr '\n' ' ' < "$SUBJECTS_FILE")
ANTS_SIF=/lustre/scratch/data/.../MainDir/fmriprep/fmriprep-25.1.1.sif


# Activate your env if build_cortex_gm_rois.py needs nibabel/numpy from there
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Run (overwrite ensures you replace the bad huge ROI files)
conda run -n glmsingle_env python -u build_cortex_rois_gm_only.py \
  --subjects $SUBJECTS \
  --ants-sif "$ANTS_SIF" \
  --gm-thr 0.30 \
  --allow-gm-only-fallback \
  --overwrite
