#!/bin/bash

# Define hyperparameters
paths=(
    "/network/scratch/j/joshi.shruti/psp/travellers/2024-09-19_23-00-51"
)
alphas=(
    "--alpha 0.0001" "--alpha 0.0005" "--alpha 0.001" "--alpha 0.01"
)
epochs=(
    "--num-epochs 700"
)
primal_lrs=(
    "--primal-lr 0.0001"
)
indicator_thresholds=(
    "--indicator-threshold 0.1"
)
norm_types=(
    "--norm_type bn" "--norm_type ln"
)
dual_lrs=(
    "--dual-lr 0.00005"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="1:00:00"
memory="32Gb"
gpu_req="gpu:1"

# Directory to store generated job scripts
mkdir -p generated_jobs

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for path in "${paths[@]}"; do
    for alpha in "${alphas[@]}"; do
        for primal_lr in "${primal_lrs[@]}"; do
            for indicator_threshold in "${indicator_thresholds[@]}"; do
                for norm_type in "${norm_types[@]}"; do
                    for dual_lr in "${dual_lrs[@]}"; do
                        for seed in "${seeds[@]}"; do
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
                            echo "python linear_sae.py ${path} ${alpha} ${primal_lr} ${dual_lr} ${norm_type} ${indicator_threshold} ${seed}" >> "${script_name}"

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



python check_wd.py /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/2024-08-22_23-37-25/13522024-08-29_13-44-02 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/13512024-08-29_13-44-02 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/2024-08-22_23-37-25/13502024-08-29_13-44-02 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/2024-08-22_23-37-25/000113502024-08-28_23-58-56 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/2024-08-22_23-37-25/000113512024-08-28_23-58-55 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/000113522024-08-29_00-14-39 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/30022024-08-29_13-44-02 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/30012024-08-29_13-44-11 /network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25/30002024-08-29_13-44-09