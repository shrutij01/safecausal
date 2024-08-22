import itertools
import subprocess

if __name__ == "__main__":
    account = "joshi.shruti"
    hyperparams = [
        [
            "/network/scratch/j/joshi.shruti/psp/gradeschooler/2024-07-08_17-03-30"
        ],
        # [
        #     "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-25_14-37-56_dgp1",
        #     "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-30_22-25-10_dgp1dense",
        #     "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-25_14-38-03_dgp2",
        #     "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-30_22-25-14_dgp2dense",
        # ],
        # ["--data-type gt_ent"],
        ["--data-type emb"],
        ["--num-epochs 1100"],
        ["--k 3"],
    ]
    print(len(hyperparams))

    init_commands = '. /home/mila/j/joshi.shruti/venvs/eqm/bin/activate && module load miniconda/3 && conda activate pytorch && export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/psp:$PYTHONPATH" && cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp'

    python_command = "python linear_sae.py"
    sbatch_command = "sbatch --gres=gpu:1 --time=1:00:0 --mem=100G"
    all_args = list(itertools.product(*hyperparams))
    print(f"Total jobs = {len(all_args)}")
    for args in all_args:
        args = " ".join(args)
        job_command = (
            sbatch_command
            + ' --wrap="'
            + init_commands
            + " && "
            + python_command
            + " "
            + args
            + '"'
        )
        print(job_command)
        result = subprocess.run(
            job_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
