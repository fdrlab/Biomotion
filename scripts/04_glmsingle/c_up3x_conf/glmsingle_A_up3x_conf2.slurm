#!/bin/bash
#SBATCH --job-name=glmsingle
#SBATCH --output=glmsingle_output_%A_%a.txt
#SBATCH --error=glmsingle_error_%A_%a.txt
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8       
#SBATCH --mem=700G               
#SBATCH --nodes=1
#SBATCH --array=1-60
#SBATCH --partition=intelsr_long
#SBATCH --mail-type=END,FAIL,TIME_LIMIT


set -euo pipefail
mkdir -p logs

# Conda
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Thread pinning to avoid oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID  ArrayIdx: $SLURM_ARRAY_TASK_ID"
which python
python -c "import slicetime; print('slicetime OK')"

srun --ntasks=1 --cpu-bind=cores --kill-on-bad-exit=1 \
  python glmsingle_A_up3x_conf.py \
    --datatype smoothed \
    --headless

