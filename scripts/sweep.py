import itertools
import subprocess

if __name__ == "__main__":
    account = "joshi.shruti"
    hyperparams = [
        [
            "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-25_14-37-56_dgp1",
            "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-30_22-25-10_dgp1dense",
            "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-25_14-38-03_dgp2",
            "/network/scratch/j/joshi.shruti/psp/toy_translator/2024-07-30_22-25-14_dgp2dense",
        ],
        ["--data-type gt_ent"],
        [
            "--model-type linv",
            "--model-type la",
            "--model-type nla --act_fxn relu",
            "--model-type nla --act_fxn leakyrelu",
            "--model-type nla --act_fxn gelu",
        ],
        ["--num-epochs 900"],
    ]

    init_commands = f'module load miniconda/3 && source ~/venvs/eqm/bin/activate && conda activate pytorch && export PYTHONPATH="/home/mila/j/joshi.shruti/causalrepl_space/psp:$PYTHONPATH" && cd /home/mila/j/joshi.shruti/causalrepl_space/psp/psp'
    python_command = "python sparse_dict.py"
    sbatch_command = f"sbatch --gres=gpu:1 --time=0:10:0 --mem=16G"
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
        subprocess.run(job_command, shell=True)
