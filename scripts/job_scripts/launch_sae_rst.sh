#!/bin/bash
# ==============================================================================
# SAE baseline training on refusal, sycophancy, and truthful-qa datasets
#
# Supports multiple SAE types: relu, topk, jumprelu
# For topk: uses --kval-topk to set k
# For all: uses --gamma-reg for regularization weight
# ==============================================================================

embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/refusal/refusal_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/refusal/refusal_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/sycophancy/sycophancy_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/sycophancy/sycophancy_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/truthful-qa/truthful-qa_gemma2_25_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/truthful-qa/truthful-qa_pythia70m_5_last_token.h5"
)

data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/refusal/refusal_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/refusal/refusal_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/sycophancy/sycophancy_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/sycophancy/sycophancy_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/truthful-qa/truthful-qa_gemma2_25_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/truthful-qa/truthful-qa_pythia70m_5_last_token.yaml"
)

# ------------------------------------------------------------------------------
# SAE architecture
# Note: --oc defaults to embedding_dim (rep_dim from YAML config) and is capped
# at embedding_dim, so we don't need to specify it explicitly.
# ------------------------------------------------------------------------------

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
    "--kval-topk 32"
)

# ------------------------------------------------------------------------------
# Regularization (--gamma-reg)
# For relu: controls L1 regularization strength on codes
# For jumprelu: controls L0 regularization strength (step function penalty)
# For topk: not used (structural sparsity, gamma * 0 = 0)
# Codes are at natural scale (no lambda scaling), so small values needed
# ------------------------------------------------------------------------------
gamma_regs=(
    "--gamma-reg 1e-6"
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
renorm_epochs=(
    "--renorm-epochs 50"
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
job_name="sae_rst"
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
            for batch_size in "${batch_sizes[@]}"; do
                for loss_type in "${loss_types[@]}"; do
                    for epochs in "${num_epochs[@]}"; do
                        for warmup in "${warmup_iters[@]}"; do
                            for wd in "${weight_decays[@]}"; do
                                for gc in "${grad_clips[@]}"; do
                                    for renorm_epoch in "${renorm_epochs[@]}"; do
                                        for seed in "${seeds[@]}"; do

                                            # Shared base flags (oc defaults to embedding_dim)
                                            base_flags="${embedding_file} ${data_config} --quick ${lr} ${loss_type} ${sae_type} ${batch_size} ${epochs} ${warmup} ${wd} ${gc} ${renorm_epoch} ${seed}"

                                            if [[ "$sae_type" == *"topk"* ]]; then
                                                for kval in "${kval_topk_values[@]}"; do
                                                    script_name="generated_jobs/job_sae_rst_${counter}.sh"
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

python -m ssae.sae ${base_flags} ${kval}
EOF
                                                    chmod +x "${script_name}"
                                                    sbatch "${script_name}"
                                                    ((counter++))
                                                done
                                            else
                                                for gamma_reg in "${gamma_regs[@]}"; do
                                                    script_name="generated_jobs/job_sae_rst_${counter}.sh"
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

python -m ssae.sae ${base_flags} ${gamma_reg}
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
