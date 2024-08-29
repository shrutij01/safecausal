#!/bin/bash

# Define hyperparameters
paths=(
    "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-30_22-25-14_dgp2dense"
    "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-25_14-37-56_dgp1"
)
data_types=(
    "--data-type gt_ent"
)
epochs=(
    "--num-epochs 700"
)
ks=(
    "--k 6"
)
lrs=(
    "--lr 0.0001"
)
alphas=(
    "--alpha 0.0001"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2"
)

# Job settings
job_name="test"
output="job_output_%j.txt"  # %j will be replaced by the job ID
error="job_error_%j.txt"
time_limit="1:00:00"
memory="16Gb"
gpu_req="gpu:1"

# Directory to store generated job scripts
mkdir -p generated_jobs

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for path in "${paths[@]}"; do
    for data_type in "${data_types[@]}"; do
        for epoch in "${epochs[@]}"; do
            for k in "${ks[@]}"; do
                for seed in "${seeds[@]}"; do
                    for lr in "${lrs[@]}"; do
                        for alpha in "${alphas[@]}"; do
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
                            echo "python linear_sae.py ${path} ${data_type} ${epoch} ${lr} ${k} ${alpha} ${seed}" >> "${script_name}"

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
