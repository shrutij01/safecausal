#!/bin/bash

# Define hyperparameters
embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/eng-french/L_32_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_31_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_21_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_15_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_13_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_5_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_3_M_llama3eng-french.h5"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_1_M_llama3eng-french.h5"
)
data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/eng-french/L_32eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_31eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_21eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_15eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_13eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_5eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_3eng-french.yaml"
    # "/network/scratch/j/joshi.shruti/ssae/eng-french/L_1eng-french.yaml"
)
overcompleteness_factors=(
    "--overcompleteness-factor 1"
)
scheduler_epochs=(
    "--scheduler-epochs 1000"
)
target_sparse_levels=(
    "--target-sparse-level 0.1"
)
batch_sizes=(
    "--batch-size 64"
)
norm_types=(
    "--norm-type ln"
)
loss_types=(
    "--loss-type absolute"
)
primal_lrs=(
    "--primal-lr 0.0005"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 5" "--seed 7"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="0:15:00"
memory="32Gb"
gpu_req="gpu:1"

# Directory to store generated job scripts
mkdir -p generated_jobs

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for idx in "${!embedding_files[@]}"; do
    embedding_file="${embedding_files[$idx]}"
    data_config="${data_configs[$idx]}"

    for target_sparse_level in "${target_sparse_levels[@]}"; do
        for primal_lr in "${primal_lrs[@]}"; do
            for overcompleteness_factor in "${overcompleteness_factors[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for norm_type in "${norm_types[@]}"; do
                        for loss_type in "${loss_types[@]}"; do
                            for scheduler_epoch in "${scheduler_epochs[@]}"; do
                                for seed in "${seeds[@]}"; do
                                    # Define a script name
                                    script_name="generated_jobs/job_${counter}.sh"

                                    # Create a batch script for each job
                                    echo "#!/bin/bash" > "${script_name}"
                                    echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                                    echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                                    echo "#SBATCH --mem=${memory}" >> "${script_name}"
                                    echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                                    echo "" >> "${script_name}"
                                    echo "module load python/3.10" >> "${script_name}"
                                    echo "source /network/scratch/j/joshi.shruti/venvs/llm/bin/activate" >> "${script_name}"
                                    echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/steeragents:/home/mila/j/joshi.shruti/causalrepl_space/cooper/:$PYTHONPATH\"" >> "${script_name}"
                                    echo "cd /home/mila/j/joshi.shruti/causalrepl_space/steeragents/ssae" >> "${script_name}"
                                    echo "python linear_sae.py --embeddings-file ${embedding_file} --data-config-file ${data_config} ${overcompleteness_factor} ${primal_lr} ${loss_type} ${norm_type} ${target_sparse_level} ${batch_size} ${scheduler_epoch} ${seed}" >> "${script_name}"

                                    # Make the script executable
                                    chmod +x "${script_name}"

                                    # Submit the job
                                    sbatch "${script_name}"

                                    # Increment counter
                                    ((counter++))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

