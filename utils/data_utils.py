import torch

from datasets import load_dataset as hf_load_dataset

from data import base

import numpy as np
import random
from random import shuffle

import os
import json
import h5py
import pandas as pd
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensorify(np_array, device):
    return torch.from_numpy(np_array).to(device).type(torch.float32)


def numpify(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labeled_sentences():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(
        script_dir,
        "..",
        "data",
        "labeled-sentences",
        "labeled_sentences_large_deduped_train.jsonl",
    )
    if not os.path.exists(datapath):
        raise FileNotFoundError()

    # Load all labeled sentences
    labeled_sentences = load_jsonl(datapath)

    cfc_tuples = []
    all_labels = []

    values_ordered = [
        "tense-present",
        "tense-past",
        "voice-active",
        "voice-passive",
        "domain-science",
        "domain-fantasy",
        "domain-news",
        "domain-other",
        "reading-level-high",
        "reading-level-low",
        "sentiment-positive",
        "sentiment-neutral",
        "sentiment-negative",
    ]

    # Create binary labels for each sentence
    sentences_with_binary_labels = []
    for data in labeled_sentences:
        sentence = data["sentence"]
        binary_labels = {}

        # Initialize all binary labels to False
        for value in values_ordered:
            binary_labels[value] = False

        # Set binary labels based on attributes
        for key, val in data.items():
            if key == "sentence":
                continue
            if key == "reading_level":
                binary_labels["reading-level-high"] = val > 11.5
                binary_labels["reading-level-low"] = val <= 11.5
            else:
                kv = f"{key}-{val}"
                if kv in binary_labels:
                    binary_labels[kv] = True

        sentences_with_binary_labels.append((sentence, binary_labels))

    # Create random pairs
    for i in range(len(sentences_with_binary_labels)):
        for j in range(i + 1, min(i + 3, len(sentences_with_binary_labels))):
            sent1, labels1 = sentences_with_binary_labels[i]
            sent2, labels2 = sentences_with_binary_labels[j]

            cfc_tuples.append([sent1, sent2])

            # Create labels indicating differences between the pair
            pair_labels = {}
            for key in values_ordered:
                pair_labels[key] = (
                    labels1[key] != labels2[key]
                )  # True if different
            all_labels.append(pair_labels)

    return cfc_tuples, all_labels


def load_dataset(
    dataset_name,
    num_samples,
    split=0.9,
    return_labels=False,
):
    if dataset_name == "truthful-qa":
        hf_dataset_name = "truthfulqa/truthful_qa"
        dataset_params = {
            "split": "validation",
            "name": "multiple_choice",
        }
        data = hf_load_dataset(hf_dataset_name, **dataset_params)
        # mc1_targets have a single correct choice and mc2_targets have
        # multiple choices that can be correct
        instruction = "True or False: "
        cfc_tuples = []
        for item in data:
            question = item["question"]
            correct_answer_index = item["mc1_targets"]["labels"].index(1)
            correct_answer_choice = item["mc1_targets"]["choices"][
                correct_answer_index
            ]
            wrong_answers_indices = [
                index
                for index, value in enumerate(item["mc1_targets"]["labels"])
                if value == 0
            ]
            wrong_answer_index = np.random.choice(wrong_answers_indices)

            # Randomly select a wrong answer
            wrong_answer_choice = item["mc1_targets"]["choices"][
                wrong_answer_index
            ]

            # Create the tuple
            correct_pair = f"{question} {instruction} {correct_answer_choice}"
            wrong_pair = f"{question} {instruction} {wrong_answer_choice}"
            cfc_tuples.append([wrong_pair, correct_pair])
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None
        cfc_train_labels = None
        cfc_test_labels = None
    elif dataset_name == "sycophancy" or dataset_name == "refusal":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sycopath = os.path.join(
            script_dir, "..", "data", dataset_name + ".json"
        )
        if not os.path.exists(sycopath):
            raise FileNotFoundError(
                f"Please download the dataset and place it in the data directory."
            )
        syco_data = load_json(sycopath)
        instruction = "Is the following answer aligned with the user's behavior? Answer only with the option: '(A) Agree' or '(B) Disagree'.\n"
        cfc_tuples = []
        for item in syco_data:
            question = item["question"]
            matching_answer = item["answer_matching_behavior"]
            not_matching_answer = item["answer_not_matching_behavior"]

            # Create paired samples: question + each answer choice
            matching_pair = f"{question}\n{matching_answer}"
            not_matching_pair = f"{question}\n{not_matching_answer}"

            # Add as [not_matching, matching] to follow your reference format
            cfc_tuples.append([not_matching_pair, matching_pair])

        # Split into train/test
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None
    elif dataset_name == "safearena":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        safepath = os.path.join(script_dir, "..", "data", "safe.json")
        harmpath = os.path.join(script_dir, "..", "data", "harm.json")
        if not os.path.exists(safepath) or not os.path.exists(harmpath):
            raise FileNotFoundError(
                f"Please download the dataset from https://huggingface.co/datasets/McGill-NLP/safearena and place it in the data directory."
            )
        safe_data = load_json(safepath)
        harm_data = load_json(harmpath)
        safe_intents = [entry["intent"] for entry in safe_data]
        harm_intents = [entry["intent"] for entry in harm_data]
        cfc_tuples = [
            [safe, harm] for safe, harm in zip(safe_intents, harm_intents)
        ]
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None
    elif dataset_name == "wildjailbreak":
        paired_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "generation_outputs/"
            "generate_vanilla_wildbreak_make_safe_meta-llama-Llama-3.1-8B-Instruct_0.6_2025-04-29-01:48:25_90",
            "paired_samples.jsonl",
        )
        if not os.path.exists(paired_file_path):
            raise FileNotFoundError(
                f"Please generate the safe-adversarial sample pairs first!"
            )
        paired_data = load_jsonl(paired_file_path)
        cfc_tuples = [
            [
                d["instruction"],
                d["model_output"]
                .split("Original harmful prompt:", 1)[1]
                .strip(),
            ]
            for d in paired_data
            if "Original harmful prompt:" in d["model_output"]
        ]

        skipped_entries = [
            {
                "instruction": d.get("instruction", ""),
                "model_output": d["model_output"],
                "reason": "Missing 'Original harmful prompt:' marker",
            }
            for d in paired_data
            if "Original harmful prompt:" not in d["model_output"]
        ]
        with open(
            "wildjailbreak_skipped_prompts.jsonl", "w", encoding="utf-8"
        ) as f:
            for entry in skipped_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None

    elif dataset_name in [
        "eng-french",
        "eng-german",
        "masc-fem-eng",
        "masc-fem-mixed",
        "2-binary",
        "corr-binary",
    ]:
        split_index = int(split * num_samples)
        cfc_train_tuples, cfc_test_tuples, concept_labels_test = (
            generate_counterfactual_pairs(
                dataset_name, split_index, num_samples
            )
        )
        cfc_train_labels = None
        cfc_test_labels = None
    else:
        raise NotImplementedError
    if concept_labels_test is not None:
        directory_location = "/network/scratch/j/joshi.shruti/ssae/"
        directory_name = os.path.join(directory_location, str(dataset_name))
        if not os.path.exists(directory_name):
            print(f"📁 Directory does not exist. Creating: {directory_name}")
            os.makedirs(directory_name)
        concept_labels_test_path = os.path.join(
            directory_name,
            "concept_labels_test_.json",
        )
        if os.path.exists(concept_labels_test_path):
            print(
                f"⚠️ Test labels already exist at: {concept_labels_test_path}. Skipping write."
            )
        else:
            print(
                f"✅ Test labels do not exist yet. Writing to {concept_labels_test_path}."
            )
            with open(concept_labels_test_path, "w") as file:
                json.dump(concept_labels_test, file)
    return cfc_train_tuples, cfc_test_tuples


def generate_counterfactual_pairs(dataset_name, split_index, num_samples):
    num_train_samples = len(range(0, split_index))
    num_test_samples = len(range(split_index, num_samples))
    if dataset_name == "eng-french":
        train_data_dict = {
            "eng_objs": list(base.eng_objs)[0:split_index],
            "french_objs": list(base.french_objs)[0:split_index],
        }
        test_data_dict = {
            "eng_objs": list(base.eng_objs)[split_index:],
            "french_objs": list(base.french_objs)[split_index:],
        }
        train_pair_sampling_plan = {
            ("eng_objs", "french_objs"): num_train_samples,
        }
        test_pair_sampling_plan = {
            ("eng_objs", "french_objs", "concept_1"): num_test_samples,
        }
    elif dataset_name == "eng-german":
        train_data_dict = {
            "eng_objs": list(base.eng_objs)[0:split_index],
            "german_objs": list(base.german_objs)[0:split_index],
        }
        test_data_dict = {
            "eng_objs": list(base.eng_objs)[split_index:],
            "german_objs": list(base.german_objs)[split_index:],
        }
        train_pair_sampling_plan = {
            ("eng_objs", "german_objs"): num_train_samples,
        }
        test_pair_sampling_plan = {
            ("eng_objs", "german_objs", "concept_1"): num_test_samples,
        }
    elif dataset_name == "masc-fem-eng":
        train_data_dict = {
            "eng_masc": list(base.eng_masc)[0:split_index],
            "eng_fem": list(base.eng_fem)[0:split_index],
        }
        test_data_dict = {
            "eng_masc": list(base.eng_masc)[split_index:],
            "eng_fem": list(base.eng_fem)[split_index:],
        }
        train_pair_sampling_plan = {
            ("eng_masc", "eng_fem"): num_train_samples,
        }
        test_pair_sampling_plan = {
            ("eng_masc", "eng_fem", "concept_1"): num_test_samples,
        }
    elif dataset_name == "masc-fem-mixed":
        train_data_dict = {
            "mixed_masc": list(base.eng_masc)[0:split_index]
            + list(base.french_masc)[0:split_index],
            "mixed_fem": list(base.eng_fem)[0:split_index]
            + list(base.french_fem)[0:split_index],
        }
        test_data_dict = {
            "mixed_masc": list(base.eng_masc)[split_index:]
            + list(base.french_masc)[split_index:],
            "mixed_fem": list(base.eng_fem)[split_index:]
            + list(base.french_fem)[split_index:],
        }
        train_pair_sampling_plan = {
            ("mixed_masc", "mixed_fem"): num_train_samples,
        }
        test_pair_sampling_plan = {
            ("mixed_masc", "mixed_fem", "concept_1"): num_test_samples,
        }
    elif dataset_name == "2-binary":
        train_data_dict = {
            "eng_masc": list(base.eng_masc)[0:split_index],
            "eng_fem": list(base.eng_fem)[0:split_index],
            "french_masc": list(base.french_masc)[0:split_index],
            "french_fem": list(base.french_fem)[0:split_index],
        }
        test_data_dict = {
            "eng_masc": list(base.eng_masc)[split_index:],
            "eng_fem": list(base.eng_fem)[split_index:],
            "french_masc": list(base.french_masc)[split_index:],
            "french_fem": list(base.french_fem)[split_index:],
        }
        train_pair_sampling_plan = {
            ("eng_masc", "french_masc"): num_train_samples,
            ("eng_fem", "french_fem"): num_train_samples,
            ("eng_masc", "eng_fem"): num_train_samples,
            ("french_masc", "french_fem"): num_train_samples,
            ("eng_masc", "french_fem"): 4 * num_train_samples,
        }
        test_pair_sampling_plan = {
            ("eng_masc", "french_masc", "concept_1"): num_test_samples,
            ("eng_fem", "french_fem", "concept_1"): num_test_samples,
            ("eng_masc", "eng_fem", "concept_2"): num_test_samples,
            ("french_masc", "french_fem", "concept_2"): num_test_samples,
        }
    elif dataset_name == "corr-binary":
        train_data_dict = {
            "eng_objs": list(base.eng_objs)[0:split_index],
            "french_objs": list(base.french_objs)[0:split_index],
            "german_objs": list(base.german_objs)[0:split_index],
        }
        test_data_dict = {
            "eng_objs": list(base.eng_objs)[split_index:],
            "french_objs": list(base.french_objs)[split_index:],
            "german_objs": list(base.german_objs)[split_index:],
        }
        train_pair_sampling_plan = {
            ("eng_objs", "french_objs"): num_train_samples,
            ("eng_objs", "german_objs"): num_train_samples,
        }
        test_pair_sampling_plan = {
            ("eng_objs", "french_objs", "concept_1"): num_test_samples,
            ("eng_objs", "german_objs", "concept_3"): num_test_samples,
        }
    else:
        raise NotImplementedError
    cfc_train_tuples, _ = sample_counterfactuals(
        train_pair_sampling_plan, train_data_dict
    )
    cfc_test_tuples, concept_labels_test = sample_counterfactuals(
        test_pair_sampling_plan, test_data_dict
    )
    shuffle(cfc_train_tuples)
    return cfc_train_tuples, cfc_test_tuples, concept_labels_test


def sample_counterfactuals(plan, data_dict):
    all_pairs = []
    concept_ids = []
    for keys, n_samples in plan.items():
        if len(keys) == 2:
            src_key, tgt_key = keys
            concept_id = None
        else:
            src_key, tgt_key, concept_id = keys
        src_list = data_dict[src_key]
        tgt_list = data_dict[tgt_key]
        max_len = min(len(src_list), len(tgt_list))
        n_samples = min(n_samples, max_len)
        indices = random.sample(range(max_len), n_samples)

        for idx in indices:
            src_word = src_list[idx]
            tgt_word = tgt_list[idx]

            if src_key == tgt_key and src_word == tgt_word:
                continue
            # Append the concept_id as part of the tuple
            if len(keys) == 3:
                all_pairs.append([src_word, tgt_word])
                concept_ids.append(concept_id)
            else:
                all_pairs.append([src_word, tgt_word])
    return all_pairs, concept_ids


def load_test_data(datafile):
    with h5py.File(datafile, "r") as f:
        cfc_embeddings_test = np.array(f["cfc_test"])
        tilde_z = cfc_embeddings_test[:, 1]
        z = cfc_embeddings_test[:, 0]
    return tilde_z, z


def load_oodprobe_paired_samples(data_dir, num_samples=1000, train_split=0.9, seed=42):
    """
    Load CSV files from oodprobe directory and create paired samples with different targets.
    Returns same format as truthful_qa for compatibility with store_embeddings.py.

    Args:
        data_dir: Path to directory containing oodprobe CSV files
        num_samples: Number of paired samples to create per file
        train_split: Fraction of samples for training (default 0.9)
        seed: Random seed for reproducibility

    Returns:
        tuple: (cfc_train_tuples, cfc_test_tuples) where each is a list of [prompt1, prompt2]
    """
    random.seed(seed)
    np.random.seed(seed)

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_pairs = []

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")

        # Load CSV file
        df = pd.read_csv(csv_file)

        # Ensure we have the required columns
        if not all(col in df.columns for col in ['prompt', 'target']):
            print(f"Skipping {csv_file.name}: missing 'prompt' or 'target' columns")
            continue

        # Group by target value
        target_groups = df.groupby('target')
        target_values = list(target_groups.groups.keys())

        if len(target_values) < 2:
            print(f"Skipping {csv_file.name}: only one target value found")
            continue

        # Create paired samples with different targets
        for _ in range(num_samples):
            # Randomly select two different target values
            target1, target2 = random.sample(target_values, 2)

            # Get random samples from each target group
            group1 = target_groups.get_group(target1)
            group2 = target_groups.get_group(target2)

            prompt1 = random.choice(group1['prompt'].tolist())
            prompt2 = random.choice(group2['prompt'].tolist())

            # Store as [prompt1, prompt2] matching truthful_qa format
            all_pairs.append([prompt1, prompt2])

    # Shuffle all pairs to mix data from different files
    random.shuffle(all_pairs)

    # Split into train/test
    split_idx = int(len(all_pairs) * train_split)
    cfc_train_tuples = all_pairs[:split_idx]
    cfc_test_tuples = all_pairs[split_idx:]

    print(f"Total created: {len(cfc_train_tuples)} train pairs and {len(cfc_test_tuples)} test pairs")

    return cfc_train_tuples, cfc_test_tuples


def save_oodprobe_pairs(data_dir, output_file, num_samples=1000, train_split=0.9, seed=42):
    """
    Create and save oodprobe paired samples to a JSON file for reproducibility.

    Args:
        data_dir: Path to directory containing oodprobe CSV files
        output_file: Path to save the paired samples JSON file
        num_samples: Number of paired samples to create per file
        train_split: Fraction of samples for training
        seed: Random seed for reproducibility
    """
    cfc_train_tuples, cfc_test_tuples = load_oodprobe_paired_samples(
        data_dir, num_samples, train_split, seed
    )

    # Save to JSON file
    output_data = {
        "train_pairs": cfc_train_tuples,
        "test_pairs": cfc_test_tuples,
        "metadata": {
            "num_samples_per_file": num_samples,
            "train_split": train_split,
            "seed": seed,
            "total_train": len(cfc_train_tuples),
            "total_test": len(cfc_test_tuples)
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved paired samples to {output_file}")
    return output_file
