#!/bin/bash

# Define hyperparameters
paths=(
    "/network/scratch/j/joshi.shruti/psp/gradeschooler/2024-07-08_17-03-30"
)

data_types=(
    "--data-type emb"
)

epochs=(
    "--num-epochs 1100"
)

ks=(
    "--k 3"
)

seeds=(
    "--seed 0", "--seed 1", "--seed 2",
)

# Job settings
job_name="test"
output="job_output.txt"
error="job_error.txt"
time_limit="1:00:00"
memory="100Gb"
gpu_req="gpu:1"

# Python command
python_command="python linear_sae.py"

# Initial setup commands
init_commands="source /home/mila/j/joshi.shruti/venvs/eqm/bin/activate && module load miniconda/3 && conda activate pytorch && export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/psp:$PYTHONPATH\" && cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp"

# Iterate over all combinations of hyperparameters
for path in "${paths[@]}"; do
    for data_type in "${data_types[@]}"; do
        for epoch in "${epochs[@]}"; do
            for k in "${ks[@]}"; do
                for seed in "${seeds[@]}"; do
                    # Construct the sbatch command
                    sbatch_command="sbatch --job-name=${job_name} --output=${output} --error=${error} --time=${time_limit} --mem=${memory} --gres=${gpu_req} --wrap=\"${init_commands} && ${python_command} ${path} ${data_type} ${epoch} ${k} ${seed}\""

                    # Echo command to terminal (for debugging)
                    echo "Submitting job: ${sbatch_command}"

                    # Submit the job
                    eval "${sbatch_command}"
            done
        done
    done
done