#!/bin/bash
#SBATCH --job-name=tfce_qc
#SBATCH --output=tfce_qc_output_%j.txt
#SBATCH --error=tfce_qc_error_%j.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_long
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=1-29

unset SLURM_EXPORT_ENV
set -euo pipefail

# Conda env
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


SUBS=(105 106 107 112 114 117 118 120 122 135 136 142 144 146 147 150 151 152 153 154 155 
157 164 165 171 172 174 175 176) # placebo participants


IDX=$((SLURM_ARRAY_TASK_ID-1))
S=${SUBS[$IDX]}


conda run -n glmsingle_env python -u /lustre/scratch/data/.../nilearn_firstlevel/one-sample_t_placebo_biomotion_vs_scrambled.py \
  --branch A_up3x \
  --subject "sub-${S}" \
  --datatype unsmoothed \
  --outdir /lustre/scratch/data/.../nilearn_firstlevel/GLMsingle_new/smooth-6 \
  --design-order-json /lustre/scratch/data/.../GLMsingle/design_order_map.json \
  --contrast-mode mean --side one_pos --contrast-sign pos \
  --smooth --smooth-fwhm 6