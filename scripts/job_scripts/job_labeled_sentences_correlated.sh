#!/bin/bash

# Define hyperparameters for correlated labeled sentences
corr_embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.1_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.2_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.5_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.9_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.99_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_1.0_gemma2_25_last_token.h5"
)

corr_data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.1_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.2_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.5_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.9_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_0.99_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/labeled-sentences-correlated/labeled_sentences_corr_ds-sp_1.0_gemma2_25_last_token.yaml"
)

# Same parameters as job_labeled_sentences.sh
encoding_dims=(
    "--oc 4096"    # overcompleteness factor
)
schedules=(
    "--schedule 3000"   # scheduler-epochs
)
targets=(
    "--target 0.005" # target-sparsity
)
batch_sizes=(
    "--batch 1024"       # batch-size - increased for better GPU utilization
)
norm_types=(
    "--norm ln"         # norm-type
)
loss_types=(
    "--loss absolute" # loss-type
)
learning_rates=(
    "--lr 0.0005"      # primal-lr
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 5" "--seed 7"
)

# New optimized parameters from refactored code
renorm_epochs=(
    "--renorm-epochs 50"    # renormalization frequency for the decoder columns
)
num_epochs=(
    "--epochs 15000"
)

# Job settings
job_name="ssae_correlated"
output="job_output_%j.txt"
error="job_error_%j.txt"
time_limit="3:00:00"
cpu_req="cpus-per-task=8"
memory="32Gb"
gpu_req="gpu:1"

# Directory to store generated job scripts
mkdir -p generated_jobs
mkdir -p logs

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for idx in "${!corr_embedding_files[@]}"; do
    embedding_file="${corr_embedding_files[$idx]}"
    data_config="${corr_data_configs[$idx]}"

    for target in "${targets[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for oc in "${encoding_dims[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for norm_type in "${norm_types[@]}"; do
                        for loss_type in "${loss_types[@]}"; do
                            for schedule in "${schedules[@]}"; do
                                for renorm_epoch in "${renorm_epochs[@]}"; do
                                    for epochs in "${num_epochs[@]}"; do
                                        for seed in "${seeds[@]}"; do
                                            # Define a script name
                                            script_name="generated_jobs/job_corr_${counter}.sh"

                                            # Create a batch script for each job
                                            echo "#!/bin/bash" > "${script_name}"
                                            echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                                            echo "#SBATCH --error=logs/job_%j.err" >> "${script_name}"
                                            echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                                            echo "#SBATCH --mem=${memory}" >> "${script_name}"
                                            echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                                            echo "#SBATCH --${cpu_req}" >> "${script_name}"
                                            echo "module load python/3.10" >> "${script_name}"
                                            echo "module load cuda/12.6.0/cudnn" >> "${script_name}"
                                            echo "source /home/mila/j/joshi.shruti/venvs/agents/bin/activate" >> "${script_name}"
                                            echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/safecausal:\$PYTHONPATH\"" >> "${script_name}"
                                            echo "cd /home/mila/j/joshi.shruti/causalrepl_space/safecausal/ssae" >> "${script_name}"

                                            # Updated command with new parameter names and optimizations
                                            echo "python ssae.py ${embedding_file} ${data_config} --quick ${oc} ${lr} ${loss_type} ${norm_type} ${target} ${batch_size} ${schedule} ${renorm_epoch} ${epochs} ${seed}" >> "${script_name}"

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