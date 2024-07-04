import torch

from datasets import load_dataset

from data import ana, gradeschooler


def load_dataset(dataset_name, **kwargs):
    if dataset_name == "ana":
        data = ana.generate_data(**kwargs)
        instruction = "Swap the letters in this string."
        cfc1_tuples, cfc2_tuples = data[:, 0], data[:, 1]
    elif dataset_name == "gradeschooler":
        cfc1_tuples = []
        cfc2_tuples = []
        file_path = kwargs.get("file_path")
        if file_path is None:
            raise ValueError(
                "file_path must be provided as a keyword argument"
            )
        with open(file_path, "r") as f:
            context_pairs = [
                line.strip().split("\t") for line in f if line.strip()
            ]
        for cp in context_pairs:
            cfc1_tuples.append(cp[0].split(",")[0])
            cfc2_tuples.append(cp[0].split(",")[1].lstrip())
        instruction = "Change any one, or two, or all three of the entities in the string, and mention how many and which you changed."
    else:
        if dataset_name == "truthful_qa":
            dataset_params = {
                "split": "validation",
                "name": "multiple_choice",
            }
        data = load_dataset(dataset_name, **dataset_params)
        instruction = "Label as 0 for False and 1 for True."
        true_targets = [
            value
            for is_true, value in zip(
                [d == 1 for d in data[0]["mc1_targets"]["labels"]],
                data[0]["mc1_targets"]["choices"],
            )
            if is_true
        ]
        false_targets = [
            value
            for is_true, value in zip(
                [d == 0 for d in data[0]["mc1_targets"]["labels"]],
                data[0]["mc1_targets"]["choices"],
            )
            if is_true
        ]
    return cfc1_tuples, cfc2_tuples, instruction


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append(
            [str(instruction) + " " + str(context)]
        )
    return instruction_plus_contexts


def pearson_corr(x, y):
    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.mm(xm.t(), ym)
    r_den = torch.sqrt(
        torch.sum(xm**2, dim=0).view(-1, 1)
        * torch.sum(ym**2, dim=0).view(1, -1)
    )
    r = r_num / r_den
    return r


def mcc_metric(Z, X):
    """
    Calculate Metric for Correlation of Correlations (MCC)
    Z: Latent variables (N, L) where N is number of samples, L is number of latent dims
    X: Observed features (N, F) where N is number of samples, F is number of features
    """
    corr_matrix = pearson_corr(Z, X)

    # Calculate correlation of the absolute values of correlation coefficients
    abs_corr_matrix = torch.abs(corr_matrix)
    upper_triangle_indices = torch.triu_indices(
        abs_corr_matrix.size(0), abs_corr_matrix.size(1), offset=1
    )
    mcc = torch.mean(
        abs_corr_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
    )

    return mcc
