#!/bin/bash

# Define hyperparameters for bias-in-bios dataset training
embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_pythia70m_5_last_token.h5"
)

data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_pythia70m_5_last_token.yaml"
)

# Updated parameter names to match refactored code
encoding_dims=(
    "--oc 4096"    # overcompleteness factor
)
schedules=(
    "--schedule 3000" #"--schedule 5000"   # scheduler-epochs
)
# Target sparsity
# - For l1: average absolute activation per (sample, feature) pair
# - For step_l0: fraction of active features (0.05 = 5% of 4096 = ~200 features)
targets_l1=(
    "--target 0.005"
)
targets_step_l0=(
    "--target 0.05"
)
batch_sizes=(
    "--batch 1024" # "--batch 512"      # batch-size - increased for better GPU utilization
)
norm_types=(
    "--norm ln"         # norm-type
)
loss_types=(
    "--loss absolute" # loss-type "--loss absolute"
)
learning_rates=(
    "--lr 0.0005" # "--lr 0.0007"      # primal-lr
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 5" "--seed 7"
)

# New optimized parameters from refactored code
renorm_epochs=(
    "--renorm-epochs 50"    # renormalization frequency for the decoder columns
)
num_epochs=(
    "--epochs 20000"
)

# Dual optimizer settings
dual_optims=(
    "--dual-optim extra-adam"
)
dual_lr_divs=(
    # "--dual-lr-div 2.0"
    "--dual-lr-div 10.0"
)

# Sparsity constraint type
# - l1: constrains average absolute activation (current default)
# - step_l0: constrains fraction of active features via differentiable step function
sparsity_types=(
    "--sparsity-type l1"
    "--sparsity-type step_l0"
)

# step_l0-specific parameters (only used when --sparsity-type step_l0)
step_l0_thresholds=(
    "--step-l0-threshold 0.01"
)
step_l0_bandwidths=(
    "--step-l0-bandwidth 0.001"
)

# Job settings
job_name="ssae_bias_in_bios"
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

# ==============================================================================
# Generate and submit jobs
# ==============================================================================

submit_job() {
    # Args: all the CLI flags as a single string
    local job_flags="$1"
    local script_name="generated_jobs/job_bias_in_bios_${counter}.sh"

    cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_${counter}
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=${time_limit}
#SBATCH --mem=${memory}
#SBATCH --gres=${gpu_req}
#SBATCH --${cpu_req}

module load python/3.10
module load cuda/12.6.0/cudnn
source /home/mila/j/joshi.shruti/venvs/agents/bin/activate
export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/safecausal:\$PYTHONPATH"
cd /home/mila/j/joshi.shruti/causalrepl_space/safecausal

python -m ssae.ssae ${job_flags}
EOF

    chmod +x "${script_name}"
    sbatch "${script_name}"
    ((counter++))
}

for idx in "${!embedding_files[@]}"; do
    embedding_file="${embedding_files[$idx]}"
    data_config="${data_configs[$idx]}"

    for lr in "${learning_rates[@]}"; do
        for oc in "${encoding_dims[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for norm_type in "${norm_types[@]}"; do
                    for loss_type in "${loss_types[@]}"; do
                        for schedule in "${schedules[@]}"; do
                            for renorm_epoch in "${renorm_epochs[@]}"; do
                                for dual_optim in "${dual_optims[@]}"; do
                                    for dual_lr_div in "${dual_lr_divs[@]}"; do
                                        for epochs in "${num_epochs[@]}"; do
                                            for sparsity_type in "${sparsity_types[@]}"; do

                                                # Select target array based on sparsity type
                                                if [[ "$sparsity_type" == *"step_l0"* ]]; then
                                                    target_list=("${targets_step_l0[@]}")
                                                else
                                                    target_list=("${targets_l1[@]}")
                                                fi

                                                for target in "${target_list[@]}"; do
                                                    for seed in "${seeds[@]}"; do

                                                        base_flags="${embedding_file} ${data_config} --quick ${oc} ${lr} ${loss_type} ${norm_type} ${target} ${batch_size} ${schedule} ${renorm_epoch} ${dual_optim} ${dual_lr_div} ${epochs} ${sparsity_type} ${seed}"

                                                        if [[ "$sparsity_type" == *"step_l0"* ]]; then
                                                            for sl0_th in "${step_l0_thresholds[@]}"; do
                                                                for sl0_bw in "${step_l0_bandwidths[@]}"; do
                                                                    submit_job "${base_flags} ${sl0_th} ${sl0_bw}"
                                                                done
                                                            done
                                                        else
                                                            submit_job "${base_flags}"
                                                        fi

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
        done
    done
done

echo "Submitted ${counter} jobs"