#!/bin/bash

# Define hyperparameters for oodprobe datasets
embedding_files=(
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/hist_fig_is_male_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/hist_fig_is_american_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/wikidata_occupation_politician_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/living_room_data_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/social_security_data_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/control_group_data_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/glue_cola_pythia70m_5_last_token.h5"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/glue_qnli_pythia70m_5_last_token.h5"
)

data_configs=(
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/hist_fig_is_male_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/hist_fig_is_american_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/wikidata_occupation_politician_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/living_room_data_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/social_security_data_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/control_group_data_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/glue_cola_pythia70m_5_last_token.yaml"
    "/network/scratch/j/joshi.shruti/ssae/oodprobe/glue_qnli_pythia70m_5_last_token.yaml"
)

oc_target_pairs=(
    "--oc 1 --target 11"
    "--oc 10 --target 0.1"
)
schedules=(
    "--schedule 3000" "--schedule 5000"
)
batch_sizes=(
    "--batch 64"
)
norm_types=(
    "--norm ln"
)
loss_types=(
    "--loss relative"
)
learning_rates=(
    "--lr 0.0005"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 5" "--seed 7"
)

renorm_epochs=(
    "--renorm-epochs 50"
)
use_amp=(
    "--use-amp"
)
num_epochs=(
    "--epochs 20000"
)

job_name="ssae_oodprobe"
output="job_output_%j.txt"
error="job_error_%j.txt"
time_limit="5:00:00"
memory="32Gb"
gpu_req="gpu:1"

mkdir -p generated_jobs
mkdir -p logs

counter=0

for idx in "${!embedding_files[@]}"; do
    embedding_file="${embedding_files[$idx]}"
    data_config="${data_configs[$idx]}"

    for oc_target in "${oc_target_pairs[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for norm_type in "${norm_types[@]}"; do
                    for loss_type in "${loss_types[@]}"; do
                        for schedule in "${schedules[@]}"; do
                            for renorm_epoch in "${renorm_epochs[@]}"; do
                                for amp in "${use_amp[@]}"; do
                                    for epochs in "${num_epochs[@]}"; do
                                        for seed in "${seeds[@]}"; do
                                            script_name="generated_jobs/job_oodprobe_${counter}.sh"

                                            echo "#!/bin/bash" > "${script_name}"
                                            echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                                            echo "#SBATCH --error=logs/job_%j.err" >> "${script_name}"
                                            echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                                            echo "#SBATCH --mem=${memory}" >> "${script_name}"
                                            echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                                            echo "module load python/3.10" >> "${script_name}"
                                            echo "module load cuda/12.6.0/cudnn" >> "${script_name}"
                                            echo "source /home/mila/j/joshi.shruti/venvs/agents/bin/activate" >> "${script_name}"
                                            echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/safecausal:$PYTHONPATH\"" >> "${script_name}"
                                            echo "cd /home/mila/j/joshi.shruti/causalrepl_space/safecausal/ssae" >> "${script_name}"

                                            echo "python ssae.py ${embedding_file} ${data_config} ${oc_target} ${lr} ${loss_type} ${norm_type} ${batch_size} ${schedule} ${renorm_epoch} ${amp} ${epochs} ${seed}" >> "${script_name}"

                                            chmod +x "${script_name}"

                                            sbatch "${script_name}"

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