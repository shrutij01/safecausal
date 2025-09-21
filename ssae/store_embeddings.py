from random import choice
from h5py._hl import dataset
import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import os
import yaml

import utils.data_utils as utils
from transformers import GPTNeoXForCausalLM, AutoTokenizer


ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_embeddings(
    contexts,
    model,
    tokenizer,
    layer,
    pooling_method="last_token",
):
    print(f"Processing {len(contexts)} context pairs ({len(contexts)*2} total texts)...")

    # Flatten all texts into a single list for vectorized processing
    all_texts = []
    for context in tqdm(contexts, desc="Flattening contexts"):
        all_texts.extend([context[0], context[1]])

    print(f"Tokenizing {len(all_texts)} texts...")
    # Single tokenization call for all texts
    tokens = tokenizer(
        all_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    print(f"Running forward pass through model (layer {layer}, {pooling_method} pooling)...")
    # Single forward pass for all texts
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs["hidden_states"][layer]

        # Apply attention mask to remove padding tokens
        attn_mask = tokens["attention_mask"]
        masked_hidden_states = hidden_states * attn_mask.unsqueeze(-1)

        if pooling_method == "last_token":
            # Get last token embeddings using attention mask
            seq_lengths = attn_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled_embeddings = hidden_states[
                torch.arange(batch_size), seq_lengths
            ].cpu()
        elif pooling_method == "sum":
            # Sum pooling across sequence dimension
            pooled_embeddings = masked_hidden_states.sum(dim=1).cpu()
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    print(f"Reshaping {len(pooled_embeddings)} embeddings back to {len(contexts)} pairs...")
    # Reshape back to pairs
    embeddings = []
    for i in tqdm(range(0, len(pooled_embeddings), 2), desc="Pairing embeddings"):
        embeddings.append([pooled_embeddings[i], pooled_embeddings[i + 1]])

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


def main(args):
    if args.dataset == "labeled-sentences":
        cfc_train_tuples, cfc_train_labels = utils.load_labeled_sentences()
        cfc_test_tuples = []  # No test split for labeled-sentences
        cfc_test_labels = []
    else:
        cfc_train_tuples, cfc_test_tuples = utils.load_dataset(
            dataset_name=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
        )
        cfc_train_labels = None
        cfc_test_labels = None
    if args.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model = transformers.LlamaForCausalLM.from_pretrained(
            args.model_id,
            token=ACCESS_TOKEN,
            low_cpu_mem_usage=True,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.bfloat16,  # check compatibility
        )
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            args.model_id, token=ACCESS_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif args.model_id == "EleutherAI/pythia-70m-deduped":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif args.model_id == "google/gemma-2-2b-it":
        model = transformers.GemmaForCausalLM.from_pretrained(
            args.model_id,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        tokenizer = transformers.GemmaTokenizer.from_pretrained(args.model_id)
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError
    print(f"Extracting embeddings from {len(cfc_train_tuples)} training samples...")
    cfc_train_embeddings = extract_embeddings(
        cfc_train_tuples,
        model,
        tokenizer,
        args.layer,
        args.pooling_method,
    )

    if len(cfc_test_tuples) > 0:
        print(f"Extracting embeddings from {len(cfc_test_tuples)} test samples...")
        cfc_test_embeddings = extract_embeddings(
            cfc_test_tuples,
            model,
            tokenizer,
            args.layer,
            args.pooling_method,
        )
    else:
        cfc_test_embeddings = []
    directory_location = "/network/scratch/j/joshi.shruti/ssae/"
    directory_name = os.path.join(directory_location, str(args.dataset))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if args.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model_name = "llama3"
    elif args.model_id == "google/gemma-2-2b-it":
        model_name = "gemma2"
    elif args.model_id == "EleutherAI/pythia-70m-deduped":
        model_name = "pythia70m"
    else:
        raise NotImplementedError
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
    ]:
        num_concepts = 1
    elif args.dataset in [
        "2-binary",
        "corr-binary",
    ]:
        num_concepts = 2
    elif args.dataset in ["labeled-sentences"]:
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
            "safearena",
            "wildjailbreak",
            "labeled-sentences",
        ],
        default="labeled-sentences",
    )
    parser.add_argument(
        "--model_id",
        default="EleutherAI/pythia-70m-deduped",
        choices=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "EleutherAI/pythia-70m-deduped",
            "google/gemma-2-2b-it",
        ],
    )
    parser.add_argument("--layer", type=int, default=31, choices=range(0, 33))
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
    # Llama-3 has 32 layers, CAA paper showed most effective steering around the
    # 13th layer for Llama-2 with 33 layers

    args = parser.parse_args()
    main(args)
