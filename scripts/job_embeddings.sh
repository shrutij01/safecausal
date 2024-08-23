#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb

source /home/mila/j/joshi.shruti/venvs/eqm/bin/activate
module load miniconda/3
conda activate pytorch
export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/psp:$PYTHONPATH"
cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp

python store_embeddings.py gradeschooler "meta-llama/Meta-Llama-3.1-8B-Instruct" --gradeschooler-file-path /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_17-58-21/gradeschooler.txt