#!/bin/bash

# Define hyperparameters
embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/masc-fem-eng/L_32_M_llama3masc-fem-eng.h5"
)
data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/masc-fem-eng/L_32masc-fem-eng.yaml"
)
overcompleteness_factors=(
    "--oc 10"
)
schedule_epochs=(
    "--schedule 1_000" "--schedule 2_000" "--schedule 3_000"
)
targets=(
    "--target 0.1" "--target 0.2"
)
batch_sizes=(
    "--batch 64"
)
norm_types=(
    "--norm ln"
)
n_concepts=(
    "--n-concepts 1"
)
learning_rates=(
    "--lr 0.0005"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="0:25:00"
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
        for learning_rate in "${learning_rates[@]}"; do
            for overcompleteness_factor in "${overcompleteness_factors[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for norm_type in "${norm_types[@]}"; do
                        for n_concept in "${n_concepts[@]}"; do
                            for schedule_epoch in "${schedule_epochs[@]}"; do
                                for seed in "${seeds[@]}"; do
                                    # Define a script name
                                    script_name="generated_jobs/job_${counter}.sh"

                                    # Create a batch script for each job
                                    echo "#!/bin/bash" > "${script_name}"
                                    echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                                    # echo "#SBATCH --output=${output}" >> "${script_name}"
                                    # echo "#SBATCH --error=${error}" >> "${script_name}"
                                    echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                                    echo "#SBATCH --mem=${memory}" >> "${script_name}"
                                    echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                                    echo "" >> "${script_name}"
                                    echo "module load python/3.10" >> "${script_name}"
                                    echo " module load cuda/12.6.0/cudnn" >> "${script_name}"
                                    echo "source /home/mila/j/joshi.shruti/venvs/agents/bin/activate" >> "${script_name}"
                                    echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/steeragents:$PYTHONPATH\"" >> "${script_name}"
                                    echo "cd /home/mila/j/joshi.shruti/causalrepl_space/steeragents/ssae" >> "${script_name}"
                                    echo "python ssae.py ${embedding_file} ${data_config} ${batch_size} ${learning_rate} ${overcompleteness_factor} ${n_concept} ${schedule_epoch} ${target} ${norm_type} ${seed}" >> "${script_name}"

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
