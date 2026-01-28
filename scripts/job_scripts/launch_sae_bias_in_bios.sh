#!/bin/bash

# ==============================================================================
# SAE baseline training on bias-in-bios dataset
#
# Supports multiple SAE types: relu, topk, jumprelu
# For topk: uses --kval-topk to set k
# For all: uses --gamma-reg for regularization weight
# ==============================================================================

embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_pythia70m_5_last_token.h5"
)

data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/bias-in-bios/bias-in-bios_pythia70m_5_last_token.yaml"
)

# ------------------------------------------------------------------------------
# SAE architecture
# ------------------------------------------------------------------------------
encoding_dims=(
    "--oc 4096"
)

# SAE types to train
# - relu: standard ReLU SAE with L1 regularization
# - topk: top-k sparse SAE (structural sparsity, no regularization needed)
# - jumprelu: JumpReLU with learnable thresholds and L0 regularization
sae_types=(
    "--sae-type relu"
    "--sae-type topk"
    "--sae-type jumprelu"
)

# Top-k values (only used when --sae-type topk)
# These will be combined with topk SAE type only
kval_topk_values=(
    "--kval-topk 16"
    "--kval-topk 32"
    "--kval-topk 64"
)

# ------------------------------------------------------------------------------
# Regularization (--gamma-reg)
# For relu/jumprelu: controls L1/L0 regularization strength
# For topk: not used (structural sparsity)
# -1 = auto: 0.01 for jumprelu, 0.1 otherwise
# ------------------------------------------------------------------------------
gamma_regs=(
    "--gamma-reg 0.001"
    "--gamma-reg 0.01"
)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------
batch_sizes=(
    "--batch 1024"
)
learning_rates=(
    "--lr 0.0005"
)
warmup_iters=(
    "--warmup-iters 1000"
)
weight_decays=(
    "--weight-decay 0.0"
)
grad_clips=(
    "--grad-clip 1.0"
)

# ------------------------------------------------------------------------------
# Loss and training
# ------------------------------------------------------------------------------
loss_types=(
    "--loss absolute"
)
num_epochs=(
    "--epochs 20000"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2"
)

# ------------------------------------------------------------------------------
# Job settings
# ------------------------------------------------------------------------------
job_name="sae_bias_in_bios"
time_limit="4:00:00"
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

for idx in "${!embedding_files[@]}"; do
    embedding_file="${embedding_files[$idx]}"
    data_config="${data_configs[$idx]}"

    for sae_type in "${sae_types[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for oc in "${encoding_dims[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for loss_type in "${loss_types[@]}"; do
                        for epochs in "${num_epochs[@]}"; do
                            for warmup in "${warmup_iters[@]}"; do
                                for wd in "${weight_decays[@]}"; do
                                    for gc in "${grad_clips[@]}"; do
                                        for seed in "${seeds[@]}"; do

                                            # Determine which additional args to use based on SAE type
                                            if [[ "$sae_type" == *"topk"* ]]; then
                                                # For topk: iterate over kval_topk values, no gamma_reg
                                                for kval in "${kval_topk_values[@]}"; do
                                                    script_name="generated_jobs/job_sae_bias_in_bios_${counter}.sh"

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
cd /home/mila/j/joshi.shruti/causalrepl_space/safecausal/ssae

python sae.py ${embedding_file} ${data_config} --quick ${oc} ${lr} ${loss_type} ${sae_type} ${kval} ${batch_size} ${epochs} ${warmup} ${wd} ${gc} ${seed}
EOF
                                                    chmod +x "${script_name}"
                                                    sbatch "${script_name}"
                                                    ((counter++))
                                                done
                                            else
                                                # For relu/jumprelu: iterate over gamma_reg values
                                                for gamma_reg in "${gamma_regs[@]}"; do
                                                    script_name="generated_jobs/job_sae_bias_in_bios_${counter}.sh"

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
cd /home/mila/j/joshi.shruti/causalrepl_space/safecausal/ssae

python sae.py ${embedding_file} ${data_config} --quick ${oc} ${lr} ${loss_type} ${sae_type} ${gamma_reg} ${batch_size} ${epochs} ${warmup} ${wd} ${gc} ${seed}
EOF
                                                    chmod +x "${script_name}"
                                                    sbatch "${script_name}"
                                                    ((counter++))
                                                done
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

echo "Submitted ${counter} jobs"
