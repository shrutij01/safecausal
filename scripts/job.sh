#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb


source /home/mila/j/joshi.shruti/venvs/eqm/bin/activate
module load miniconda/3
conda activate pytorch
export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/psp:$PYTHONPATH"
cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp

python sparse_dict.py "/network/scratch/j/joshi.shruti/psp/gradeschooler/2024-07-08_17-03-30" --data-type emb --model-type la --num-epochs 1100
