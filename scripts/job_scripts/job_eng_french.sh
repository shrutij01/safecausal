#!/bin/bash

# Define hyperparameters for refactored SSAE code
embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/eng-french/L_32_M_llama3eng-french.h5"
)

data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/eng-french/L_32eng-french.yaml"
)

# Updated parameter names to match refactored code
overcompletenesses=(
    "--oc 10"    # overcompleteness factor
)
schedules=(
    "--schedule 3000" "--schedule 5000"   # scheduler-epochs
)
targets=(
    "--target 0.1"      # target-sparse-level
)
batch_sizes=(
    "--batch 64"        # batch-sizeh
)
norm_types=(
    "--norm ln"         # norm-type
)
loss_types=(
    "--loss absolute"   "--loss relative" # loss-type
)
learning_rates=(
    "--lr 0.0005"       # primal-lr
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 5" "--seed 7"
)

# New optimized parameters from refactored code
renorm_epochs=(
    "--renorm-epochs 50"    # renormalization frequency for the decoder columns
)
use_amp=(
    "--use-amp"             # Enable mixed precision training
)

# Job settings
job_name="ssae_optimized"
output="job_output_%j.txt"
error="job_error_%j.txt"
time_limit="1:00:00"
cpu_req="cpus-per-task=8"
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

    for target in "${targets[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for oc in "${overcompletenesses[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for norm_type in "${norm_types[@]}"; do
                        for loss_type in "${loss_types[@]}"; do
                            for schedule in "${schedules[@]}"; do
                                for renorm_epoch in "${renorm_epochs[@]}"; do
                                    for amp in "${use_amp[@]}"; do
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
                                            echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/steeragents:$PYTHONPATH\"" >> "${script_name}"
                                            echo "cd /home/mila/j/joshi.shruti/causalrepl_space/steeragents/ssae" >> "${script_name}"

                                            # Updated command with new parameter names and optimizations
                                            echo "python ssae.py ${embedding_file} ${data_config} ${oc} ${lr} ${loss_type} ${norm_type} ${target} ${batch_size} ${schedule} ${renorm_epoch} ${amp} ${seed}" >> "${script_name}"

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
    done
done