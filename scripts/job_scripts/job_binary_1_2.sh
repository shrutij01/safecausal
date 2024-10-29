#!/bin/bash

# Define hyperparameters
embedding_files=(
    "/network/scratch/j/joshi.shruti/psp/binary_1_2/binary_1_2_embeddings_layer_32.h5"
)
data_configs=(
    "/network/scratch/j/joshi.shruti/psp/binary_1_2/binary_1_2_32_config.yaml"
)
alphas=(
    "--alpha 5" "--alpha 3" "--alpha 1"
)
epochs=(
    "--num-epochs 20000"
)
primal_lrs=(
    "--primal-lr 0.01"
)
indicator_thresholds=(
    "--indicator-threshold 0.1"
)
norm_types=(
    "--norm-type bn" "--norm-type ln"
)
dual_lrs=(
    "--dual-lr 0.005"
)
num_concepts=(
    "--num-concepts 100" # "--num-concepts 100" "--num-concepts 50"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" # "--seed 5" "--seed 7"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="0:00:00"
memory="80Gb"
gpu_req="gpu:a100:1"

# Directory to store generated job scripts
mkdir -p generated_jobs

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for embedding_file in "${embedding_files[@]}"; do
    for data_config in "${data_configs[@]}"; do
        for alpha in "${alphas[@]}"; do
            for primal_lr in "${primal_lrs[@]}"; do
                for indicator_threshold in "${indicator_thresholds[@]}"; do
                    for norm_type in "${norm_types[@]}"; do
                        for dual_lr in "${dual_lrs[@]}"; do
                            for epoch in "${epochs[@]}"; do
                                for seed in "${seeds[@]}"; do
                                    for num_concept in "${num_concepts[@]}"; do
                                        # Define a script name
                                        script_name="generated_jobs/job_${counter}.sh"

                                        # Create a batch script for each job
                                        echo "#!/bin/bash" > "${script_name}"
                                        echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                                        echo "#SBATCH --output=${output}" >> "${script_name}"
                                        echo "#SBATCH --error=${error}" >> "${script_name}"
                                        echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                                        echo "#SBATCH --mem=${memory}" >> "${script_name}"
                                        echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                                        echo "" >> "${script_name}"
                                        echo "source /home/mila/j/joshi.shruti/venvs/eqm/bin/activate" >> "${script_name}"
                                        echo "module load miniconda/3" >> "${script_name}"
                                        echo "conda activate pytorch" >> "${script_name}"
                                        echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/psp:\$PYTHONPATH\"" >> "${script_name}"
                                        echo "cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp" >> "${script_name}"
                                        echo "python linear_sae.py --embeddings-file ${embedding_file} --data-config-file ${data_config} ${alpha} ${primal_lr} ${dual_lr} ${norm_type} ${indicator_threshold} ${num_concept} ${epoch} ${seed}" >> "${script_name}"

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
