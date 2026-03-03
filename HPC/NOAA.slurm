#!/bin/bash
#SBATCH --job-name=TornadoQ_NOAA_BINARY
#SBATCH --output=./logs/TornadoQ_NOAA_BINARY_%A_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amacieju@odu.edu
#SBATCH --partition=gpu
#SBATCH --array=0-9

module load python3

data=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
pct=${data[$SLURM_ARRAY_TASK_ID]}

crun -p ~/envs/TornadoQ python -u full_pipeline_NOAA_dataset.py --percent_data "$pct"