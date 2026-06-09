#!/bin/bash
#
#SBATCH --job-name=fmriprep
#SBATCH --output=fmriprep_output_%A_%a.txt
#SBATCH --error=fmriprep_error_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=1-60
#SBATCH --partition=intelsr_medium
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

#SBATCH --export=NONE                 
unset SLURM_EXPORT_ENV           

# Exit immediately on error
set -euo pipefail



WORKSPACE_DIR="XY"

BIDS_DIR="${WORKSPACE_DIR}/MainDir/data"
OUTPUT_DIR="${WORKSPACE_DIR}/preprocessed"
BASE_WORK_DIR="${WORKSPACE_DIR}/workdir"
FS_LICENSE="${WORKSPACE_DIR}/MainDir/fmriprep/license.txt"
APPTAINER_IMAGE="${WORKSPACE_DIR}/MainDir/fmriprep/fmriprep-25.1.1.sif"
TEMPLATEFLOW_HOME="${WORKSPACE_DIR}/templateflow"

# Get subject ID using SLURM array task index
SUBJECT_LIST=( $(ls -d "${BIDS_DIR}"/sub-*) )
SUBJECT_ID=$(basename "${SUBJECT_LIST[${SLURM_ARRAY_TASK_ID}-1]}")

# This prevents jobs from crashing by writing to the same temp files.
UNIQUE_WORK_DIR="${BASE_WORK_DIR}/${SLURM_ARRAY_JOB_ID}_${SUBJECT_ID}"
mkdir -p "${UNIQUE_WORK_DIR}"

# --- Logging and job information ---
echo "Starting fMRIPrep for subject: ${SUBJECT_ID}"
echo "SLURM Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Allocated Memory: ${SLURM_MEM_PER_NODE}M"
echo "Unique Work Directory: ${UNIQUE_WORK_DIR}"

# Note the use of --nthreads instead of --nprocs for newer fMRIPrep versions
apptainer run --cleanenv \
    --env TEMPLATEFLOW_HOME=/templateflow \
    -B "${BIDS_DIR}":/data:ro \
    -B "${OUTPUT_DIR}":/out \
    -B "${UNIQUE_WORK_DIR}":/work \
    -B "${FS_LICENSE}":/fslicense.txt \
    -B "${TEMPLATEFLOW_HOME}":/templateflow \
    "${APPTAINER_IMAGE}" \
    /data /out participant \
    --participant-label "${SUBJECT_ID#sub-}" \
    -w /work \
    --fs-license-file /fslicense.txt \
    --dummy-scans 5 \
    --output-spaces MNI152NLin2009cAsym:res-1.7 \
    --ignore slicetiming \
    --write-graph \
    --nthreads ${SLURM_CPUS_PER_TASK} \
    --mem_mb 46000

echo "Finished fMRIPrep for ${SUBJECT_ID}"