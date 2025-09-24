from random import choice
import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import os
import yaml

import utils.data_utils as utils
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_embeddings(
    contexts,
    model,
    tokenizer,
    layer,
    pooling_method="last_token",
    batch_size=128,
):
    print(
        f"Processing {len(contexts)} context pairs ({len(contexts)*2} total texts) in batches of {batch_size}..."
    )

    all_embeddings = []
    num_batches = (len(contexts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(contexts))
        batch_contexts = contexts[start_idx:end_idx]

        # Flatten batch texts
        batch_texts = []
        for context in batch_contexts:
            batch_texts.extend([context[0], context[1]])

        # Tokenize batch
        tokens = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Forward pass for batch
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs["hidden_states"][layer]

            if pooling_method == "last_token":
                # For left padding, last token is always at the rightmost position
                pooled_embeddings = hidden_states[:, -1, :]
            elif pooling_method == "sum":
                # Sum pooling across sequence dimension
                pooled_embeddings = hidden_states.sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

            # Reshape to pairs and move to CPU to save GPU memory
            batch_embeddings = pooled_embeddings.view(
                -1, 2, pooled_embeddings.shape[-1]
            ).cpu()
            all_embeddings.append(batch_embeddings)

        # Clear GPU cache after each batch
        del (
            tokens,
            outputs,
            hidden_states,
            pooled_embeddings,
        )
        torch.cuda.empty_cache()

    # Concatenate all batch results
    print(f"Concatenating {len(all_embeddings)} batches...")
    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings


def store_embeddings(
    filename, cfc_train, cfc_test, cfc_train_labels=None, cfc_test_labels=None
):
    with h5py.File(filename, "w") as hdf_file:
        hdf_file.create_dataset("cfc_train", data=cfc_train)
        hdf_file.create_dataset("cfc_test", data=cfc_test)

        if cfc_train_labels is not None:
            # Convert labels to a format suitable for HDF5 storage
            import json

            train_labels_json = [
                json.dumps(labels) for labels in cfc_train_labels
            ]
            hdf_file.create_dataset("cfc_train_labels", data=train_labels_json)

        if cfc_test_labels is not None and len(cfc_test_labels) > 0:
            import json

            test_labels_json = [
                json.dumps(labels) for labels in cfc_test_labels
            ]
            hdf_file.create_dataset("cfc_test_labels", data=test_labels_json)


def load_model_and_tokenizer(model_id):
    """Load model and tokenizer based on model_id."""
    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_id,
            token=ACCESS_TOKEN,
            low_cpu_mem_usage=True,
        ).to(device)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            model_id, token=ACCESS_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "llama3"
    elif model_id == "EleutherAI/pythia-70m-deduped":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "pythia70m"
    elif model_id == "google/gemma-2-2b-it":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma2"
    else:
        raise NotImplementedError(f"Model {model_id} not supported")

    return model, tokenizer, model_name


def main(args):
    # Load model and tokenizer once at the beginning
    model, tokenizer, model_name = load_model_and_tokenizer(args.model_id)

    if args.dataset == "labeled-sentences":
        cfc_train_tuples, cfc_train_labels = utils.load_labeled_sentences()
        cfc_test_tuples = []
        cfc_test_labels = []
    elif args.dataset == "labeled-sentences-correlated":
        # Get list of correlated files to process individually
        corr_files = utils.load_labeled_sentences_correlated()

        for corr_file in corr_files:
            # Extract correlation level from filename

            filename = os.path.basename(corr_file)
            # Extract corr-ds-sp-X.X from filename
            import re

            match = re.search(r"corr_ds-sp-([0-9.]+)", filename)
            if match:
                corr_level = match.group(1)
                dataset_name = f"corr-ds-sp-{corr_level}"
            else:
                dataset_name = filename.replace(".jsonl", "")

            print(f"Processing {dataset_name}...")

            # Load data for this specific correlation file
            cfc_train_tuples, cfc_train_labels = (
                utils.load_single_correlated_file(corr_file)
            )
            cfc_test_tuples = []
            cfc_test_labels = []

            print(
                f"Extracting embeddings from {len(cfc_train_tuples)} training samples for {dataset_name}..."
            )
            cfc_train_embeddings = extract_embeddings(
                cfc_train_tuples,
                model,
                tokenizer,
                args.layer,
                args.pooling_method,
                args.batch_size,
            )

            cfc_test_embeddings = []

            # Create output directory
            directory_location = "/network/scratch/j/joshi.shruti/ssae/"
            directory_name = os.path.join(
                directory_location, "labeled-sentences-correlated"
            )
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            # Create unique filename for this correlation level
            filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.h5"
            filepath = os.path.join(directory_name, filename)
            store_embeddings(
                filepath,
                cfc_train_embeddings,
                cfc_test_embeddings,
                cfc_train_labels,
                cfc_test_labels,
            )

            config = {
                "rep_dim": cfc_train_embeddings.shape[-1],
                "dataset": dataset_name,
                "model": args.model_id,
                "layer": args.layer,
                "pooling_method": args.pooling_method,
                "data_seed": 21,
                "training_dataset_length": len(cfc_train_embeddings),
                "test_dataset_length": 0,
                "num_concepts": 5,
                "llm_layer": args.layer,
                "split": 0.9,
            }

            config_filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.yaml"
            config_filepath = os.path.join(directory_name, config_filename)
            with open(config_filepath, "w") as f:
                yaml.dump(config, f)

            print(f"Saved {dataset_name} embeddings to {filepath}")
            print(f"Saved {dataset_name} config to {config_filepath}")

        return
    elif args.dataset == "oodprobe":
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "oodprobe",
        )
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

        for csv_file in csv_files:
            dataset_name = (
                csv_file.replace(".csv", "").lower().replace(" ", "_")
            )
            print(f"Processing {dataset_name}...")

            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_csv_path = os.path.join(temp_dir, csv_file)
                shutil.copy(os.path.join(data_dir, csv_file), temp_csv_path)

                cfc_train_tuples, cfc_test_tuples = (
                    utils.load_oodprobe_paired_samples(
                        data_dir=temp_dir,
                        num_samples=args.num_samples,
                        train_split=args.split,
                        seed=42,
                    )
                )

            print(
                f"Extracting embeddings from {len(cfc_train_tuples)} training samples for {dataset_name}..."
            )
            cfc_train_embeddings = extract_embeddings(
                cfc_train_tuples,
                model,
                tokenizer,
                args.layer,
                args.pooling_method,
                args.batch_size,
            )

            if len(cfc_test_tuples) > 0:
                print(
                    f"Extracting embeddings from {len(cfc_test_tuples)} test samples for {dataset_name}..."
                )
                cfc_test_embeddings = extract_embeddings(
                    cfc_test_tuples,
                    model,
                    tokenizer,
                    args.layer,
                    args.pooling_method,
                    args.batch_size,
                )
            else:
                cfc_test_embeddings = []

            directory_location = "/network/scratch/j/joshi.shruti/ssae/"
            directory_name = os.path.join(directory_location, "oodprobe")
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.h5"
            filepath = os.path.join(directory_name, filename)
            store_embeddings(
                filepath, cfc_train_embeddings, cfc_test_embeddings
            )

            config = {
                "rep_dim": cfc_train_embeddings.shape[-1],
                "dataset": dataset_name,
                "model": args.model_id,
                "layer": args.layer,
                "pooling_method": args.pooling_method,
                "num_samples": args.num_samples,
                "split": args.split,
            }

            config_filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.yaml"
            config_filepath = os.path.join(directory_name, config_filename)

            with open(config_filepath, "w") as f:
                yaml.dump(config, f)

            print(f"Saved {dataset_name} embeddings to {filepath}")
            print(f"Saved {dataset_name} config to {config_filepath}")

        return
    else:
        cfc_train_tuples, cfc_test_tuples = utils.load_dataset(
            dataset_name=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
        )
        cfc_train_labels = None
        cfc_test_labels = None

    print(
        f"Extracting embeddings from {len(cfc_train_tuples)} training samples..."
    )
    cfc_train_embeddings = extract_embeddings(
        cfc_train_tuples,
        model,
        tokenizer,
        args.layer,
        args.pooling_method,
        args.batch_size,
    )

    if len(cfc_test_tuples) > 0:
        print(
            f"Extracting embeddings from {len(cfc_test_tuples)} test samples..."
        )
        cfc_test_embeddings = extract_embeddings(
            cfc_test_tuples,
            model,
            tokenizer,
            args.layer,
            args.pooling_method,
            args.batch_size,
        )
    else:
        cfc_test_embeddings = []

    directory_location = "/network/scratch/j/joshi.shruti/ssae/"
    directory_name = os.path.join(directory_location, str(args.dataset))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    embeddings_path = os.path.join(
        directory_name,
        str(args.dataset)
        + "_"
        + str(model_name)
        + "_"
        + str(args.layer)
        + "_"
        + str(args.pooling_method)
        + ".h5",
    )
    store_embeddings(
        embeddings_path,
        cfc_train_embeddings,
        cfc_test_embeddings,
        cfc_train_labels,
        cfc_test_labels,
    )
    if args.dataset in [
        "eng-french",
        "eng-german",
        "masc-fem-eng",
        "masc-fem-mixed",
        "truthful-qa",
        "safearena",
        "refusal",
        "sycophancy",
    ]:
        num_concepts = 1
    elif args.dataset in [
        "2-binary",
        "corr-binary",
    ]:
        num_concepts = 2
    elif args.dataset in ["labeled-sentences", "labeled-sentences-correlated"]:
        num_concepts = 5
    elif args.dataset == "categorical":
        num_concepts = 135
    elif args.dataset in ["safearena", "wildjailbreak"]:  # fixme
        num_concepts = 10
    else:
        raise NotImplementedError
    config = {
        "dataset": args.dataset,
        "training_dataset_length": len(cfc_train_embeddings),
        "test_dataset_length": len(cfc_test_embeddings),
        "rep_dim": cfc_train_embeddings[0][0].shape[0],
        "num_concepts": num_concepts,
        "model": args.model_id,
        "llm_layer": args.layer,
        "split": 0.9,
        "data_seed": 21,  # Fixed seed used for data pairing/sampling
    }
    config_path = os.path.join(
        directory_name,
        str(args.dataset)
        + "_"
        + str(model_name)
        + "_"
        + str(args.layer)
        + "_"
        + str(args.pooling_method)
        + ".yaml",
    )
    with open(config_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "truthful-qa",
            "refusal",
            "sycophancy",
            "labeled-sentences",
            "labeled-sentences-correlated",
            "oodprobe",
        ],
        default="labeled-sentences",
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-2-2b-it",
        choices=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "EleutherAI/pythia-70m-deduped",
            "google/gemma-2-2b-it",
        ],
    )
    parser.add_argument("--layer", type=int, default=25, choices=range(0, 33))
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--pooling-method",
        choices=["last_token", "sum"],
        default="last_token",
        help="Pooling method for sequence embeddings: 'last_token' or 'sum'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing contexts (default: 128, optimized for A100-class GPUs)",
    )
    # Llama-3 has 32 layers, CAA paper showed most effective steering around the
    # 13th layer for Llama-2 with 33 layers

    args = parser.parse_args()
    main(args)
