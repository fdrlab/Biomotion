#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inference_output_%A.txt
#SBATCH --error=inference_error_%A.txt
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_long
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

unset SLURM_EXPORT_ENV
set -euo pipefail
module purge

# === FSL in Apptainer wrappers (robust binding) ===
FSL_IMG="/lustre/scratch/data/.../containers/fsl_6.0.4.sif"
WRAPDIR="${SLURM_TMPDIR:-/tmp}/fslwrap"; mkdir -p "$WRAPDIR"

make_wrap () {
  local name="$1"
  cat > "${WRAPDIR}/${name}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BINDARGS=(-B /lustre:/lustre)
# Bind /scratch only if it exists on the host
if [ -d /scratch ]; then
  BINDARGS+=(-B /scratch:/scratch)
fi
# Keep container env clean so host Conda/etc. don't leak in
apptainer exec --cleanenv "${BINDARGS[@]}" "__FSL_IMG__" "$(basename "$0")" "$@"
EOF
  sed -i "s|__FSL_IMG__|${FSL_IMG}|g" "${WRAPDIR}/${name}"
  chmod +x "${WRAPDIR}/${name}"
}

for cmd in randomise randomise_parallel fslmerge fslmaths flirt; do
  make_wrap "$cmd"
done
export PATH="${WRAPDIR}:$PATH"

# === Python env ===
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Threading caps for numpy/nilearn if used inside Python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional quick smoke test (uncomment if you want to sanity-check the wrappers)
fslmaths --version || true
flirt -version || true

MASK="/lustre/scratch/data/.../searchlight_RSA/unsmoothed_cov20_max_dist5_k60/group_masks/tfce_thresholds_excl114_139/bio_rankortho_excl114_139_mask_p100_58of58_min1.nii.gz"
CSV="/lustre/scratch/data/.../unblinding_lorazepam_placebo_no_sub-159_114_or_139.csv"

export FSLOUTPUTTYPE=NIFTI_GZ

conda run -n glmsingle_env python -u /lustre/scratch/data/.../searchlight_RSA/group_tfce_two_group.py \
  --datatype unsmoothed_cov20_max_dist5_k60 \
  --predictor model6_castaldi_fvp_STD  \
  --metric crossnobis \
  --smooth-fwhm 6 \
  --permutations 10000 \
  --parallel \
  --mask "$MASK" \
  --groups-csv "$CSV" \
  --subject-col subject \
  --group-col group \
  --groupA-label lorazepam \
  --groupB-label placebo \
  --tfce-name beta_cast_fvp_adaptiveK_k60_crossnobis_s6mm_maskp100_min60_fitok