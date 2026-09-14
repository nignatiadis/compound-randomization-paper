#!/bin/bash
#SBATCH --job-name=cr-main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-ignat
#SBATCH --partition=caslake

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load R/4.4.1

export JULIA_NUM_THREADS=1
export JULIA_LOAD_PATH='@:@stdlib'
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# submit_main.sh supplies the resource requests and task group.
julia --project=. --startup-file=no --threads=1 simulations/run_simulation.jl "$1" "${SLURM_ARRAY_TASK_ID}"
