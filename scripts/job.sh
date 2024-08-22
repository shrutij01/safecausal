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
    "--seed 0" "--seed 1" "--seed 2"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="1:00:00"
memory="100Gb"
gpu_req="gpu:1"

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for path in "${paths[@]}"; do
    for data_type in "${data_types[@]}"; do
        for epoch in "${epochs[@]}"; do
            for k in "${ks[@]}"; do
                for seed in "${seeds[@]}"; do
                    # Construct the full command for the wrap, including a shebang
                    full_command="#!/bin/bash\nsource /home/mila/j/joshi.shruti/venvs/eqm/bin/activate && module load miniconda/3 && conda activate pytorch && export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/psp:\$PYTHONPATH\" && cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp && python linear_sae.py ${path} ${data_type} ${epoch} ${k} ${seed}"

                    # Construct the sbatch command
                    sbatch_command="sbatch --job-name=${job_name}_${counter} --output=${output} --error=${error} --time=${time_limit} --mem=${memory} --gres=${gpu_req} --wrap=\"$full_command\""

                    # Echo command to terminal (for debugging)
                    echo "Submitting job: $sbatch_command"

                    # Submit the job
                    eval "$sbatch_command"
                    ((counter++))
                done
            done
        done
    done
done
