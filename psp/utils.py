import torch

from datasets import load_dataset as hf_load_dataset

from data import ana, gradeschooler

import numpy as np
import concurrent.futures
from typing import List


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
    elif dataset_name == "truthful_qa":
        import ipdb

        ipdb.set_trace()
        hf_dataset_name = "truthfulqa/truthful_qa"
        dataset_params = {
            "split": "validation",
            "name": "multiple_choice",
        }
        data = hf_load_dataset(hf_dataset_name, **dataset_params)
        # mc1_targets have a single correct choice and mc2_targets have
        # multiple choices that can be correct
        instruction = "Label as 0 for False and 1 for True."
        cfc2_tuples = [
            value
            for is_true, value in zip(
                [d == 1 for d in data[0]["mc1_targets"]["labels"]],
                data[0]["mc1_targets"]["choices"],
            )
            if is_true
        ]
        cfc1_tuples = [
            value
            for is_true, value in zip(
                [d == 0 for d in data[0]["mc1_targets"]["labels"]],
                data[0]["mc1_targets"]["choices"],
            )
            if is_true
        ]
    else:
        raise NotImplementedError
    return cfc1_tuples, cfc2_tuples, instruction


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append(
            [str(instruction) + " " + str(context)]
        )
    return instruction_plus_contexts


class DisentanglementScores:
    def __init__(self):
        pass

    def compute_correlation(self, rep):
        return np.corrcoef(rep, rowvar=False)

    def get_upper_tri(self, rep):
        indices = np.triu_indices_from(rep, k=1)
        return rep[indices]

    def compute_mcc(self, out_rep, in_rep):
        return np.corrcoef(
            self.get_upper_tri(self.compute_correlation(out_rep)),
            self.get_upper_tri(self.compute_correlation(in_rep)),
        )[0, 1]

    def get_mcc_scores(self, out_reps, in_reps):
        mcc_scores = self.compute_mcc(out_reps, in_reps)
        return mcc_scores
