#!/bin/bash
#SBATCH --job-name=smoothen
#SBATCH --output=logs/smoothen_%A_%a.out
#SBATCH --error=logs/smoothen_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=800G
#SBATCH --array=1-60%10 
#SBATCH --partition=intelsr_short
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail
mkdir -p logs

# Use the same env you use for GLMsingle
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Keep threaded libs tame (gaussian_filter isn't BLAS, but be safe)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------- FIXED SUBJECT LIST ----------
SUBJECTS=(
  sub-102 sub-103 sub-105 sub-106 sub-107 sub-110 sub-112 sub-114 sub-115
  sub-116 sub-117 sub-118 sub-120 sub-121 sub-122 sub-123 sub-126 sub-127
  sub-129 sub-130 sub-131 sub-134 sub-135 sub-136 sub-137 sub-139 sub-140
  sub-142 sub-143 sub-144 sub-145 sub-146 sub-147 sub-150 sub-151 sub-152
  sub-153 sub-154 sub-155 sub-156 sub-157 sub-158 sub-159 sub-161 sub-162
  sub-164 sub-165 sub-166 sub-167 sub-168 sub-171 sub-172 sub-173 sub-174
  sub-175 sub-176 sub-177 sub-178 sub-179 sub-180
)

# Validate array index
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#SUBJECTS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} out of range 1..${#SUBJECTS[@]}"
  exit 1
fi

SUBJECT_ID="${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "Smoothing for: ${SUBJECT_ID}"
which python; python --version

# Run fast 4D smoother; no time smoothing; mask applied after
python smooth_subjects.py "${SUBJECT_ID}" --fwhm 3 --runs 01 02 --overwrite

echo "Done: ${SUBJECT_ID}"
