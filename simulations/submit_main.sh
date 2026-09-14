#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Each array runs one resource class through the same Julia runner.
sbatch --job-name=cr-k3-5 --array=1-20 --time=06:00:00 --mem=4G simulations/main.slurm k3_5
sbatch --job-name=cr-k7-9 --array=1-30 --time=06:00:00 --mem=6G simulations/main.slurm k7_9
sbatch --job-name=cr-k11 --array=1-30 --time=06:00:00 --mem=8G simulations/main.slurm k11
sbatch --job-name=cr-k13 --array=1-60 --time=10:00:00 --mem=16G simulations/main.slurm k13
