#!/bin/bash

# Define hyperparameters
paths=(
    "/network/scratch/j/joshi.shruti/psp/gradeschooler/2024-07-08_17-03-30"
)
data_types=(
    "--data-type emb"
)
epochs=(
    "--num-epochs 1100"
)
ks=(
    "--k 3"
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2"
)

# Directory for batch scripts
mkdir -p batch_scripts

# Create and submit a batch script for each combination of hyperparameters
counter=0
for path in "${paths[@]}"; do
    for data_type in "${data_types[@]}"; do
        for epoch in "${epochs[@]}"; do
            for k in "${ks[@]}"; do
                for seed in "${seeds[@]}"; do
                    script_name="batch_scripts/job_${counter}.sh"
                    cat > "$script_name" <<- EOM
                    #!/bin/bash
                    #SBATCH --job-name=test_${counter}
                    #SBATCH --output=job_output_${counter}.txt
                    #SBATCH --error=job_error_${counter}.txt
                    #SBATCH --time=0:30:00
                    #SBATCH --mem=16Gb
                    #SBATCH --gres=gpu:1

                    source /home/mila/j/joshi.shruti/venvs/eqm/bin/activate
                    module load miniconda/3
                    conda activate pytorch
                    export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/psp:\$PYTHONPATH"
                    cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp

                    python linear_sae.py $path $data_type $epoch $k $seed
EOM

                    # Submit the batch job
                    sbatch "$script_name"
                    ((counter++))
                done
            done
        done
    done
done
