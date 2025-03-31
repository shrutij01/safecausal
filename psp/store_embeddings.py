from h5py._hl import dataset
import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import os
import yaml

import psp.data_utils as utils

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_last_token_embedding(tokens, model, layer):
    """based on https://huggingface.co/castorini/repllama-v1-7b-lora-passage
    to compute sentence level embeddings
    """
    with torch.no_grad():
        last_hidden_states = model(**tokens, output_hidden_states=True)[
            "hidden_states"
        ][layer].squeeze()[-1]
    return last_hidden_states.cpu()


def extract_embeddings(
    contexts,
    model,
    tokenizer,
    layer,
):
    embeddings = []
    for context in tqdm(contexts):
        tokens_x = tokenizer(context[0], return_tensors="pt").to(device)
        tokens_tilde_x = tokenizer(context[1], return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings.append(
                [
                    get_last_token_embedding(tokens_x, model, layer),
                    get_last_token_embedding(tokens_tilde_x, model, layer),
                ]
            )
        del tokens_x
        del tokens_tilde_x
    return embeddings


def store_embeddings(filename, cfc_train, cfc_test):
    with h5py.File(filename, "w") as hdf_file:
        hdf_file.create_dataset("cfc_train", data=cfc_train)
        hdf_file.create_dataset("cfc_test", data=cfc_test)


def main(args):
    cfc_tuples = utils.load_dataset(
        dataset_name=args.dataset_name,
    )
    split = int(0.9 * len(cfc_tuples))
    cfc_train, cfc_test = cfc_tuples[0:split], cfc_tuples[split:]

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        args.model_id, token=ACCESS_TOKEN
    )
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_id,
        token=ACCESS_TOKEN,
        low_cpu_mem_usage=True,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16,  # check compatibility
    )
    cfc_train_embeddings = extract_embeddings(
        cfc_train,
        model,
        tokenizer,
        args.llm_layer,
    )
    cfc_test_embeddings = extract_embeddings(
        cfc_test,
        model,
        tokenizer,
        args.llm_layer,
    )
    directory_location = "/network/scratch/j/joshi.shruti/psp/"
    directory_name = os.path.join(directory_location, str(args.dataset_name))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    embeddings_path = os.path.join(
        directory_name,
        str(args.dataset_name)
        + "_L_"
        + str(args.llm_layer)
        + "_M_"
        + str(args.model_id)
        + ".h5",
    )
    store_embeddings(
        embeddings_path,
        cfc_train_embeddings,
        cfc_test_embeddings,
    )
    if args.dataset_name == "binary_1" or args.dataset_name == "binary_1_2":
        num_concepts = 1
    elif args.dataset_name == "binary_2" or args.dataset_name == "binary_corr":
        num_concepts = 2
    elif args.dataset_name == "truthful_qa":
        num_concepts = 1
    elif args.dataset_name == "categorical":
        num_concepts = 3
    else:
        raise NotImplementedError
    config = {
        "dataset": args.dataset_name,
        "training_dataset_length": len(cfc_train_embeddings),
        "test_dataset_length": len(cfc_test_embeddings),
        "rep_dim": cfc_train_embeddings[0][0].shape[0],
        "num_concepts": num_concepts,
        "model": args.model_id,
        "llm_layer": args.llm_layer,
        "split": 0.9,
    }
    config_path = os.path.join(
        directory_name,
        str(args.dataset_name) + "_" + str(args.llm_layer) + "_config.yaml",
    )
    with open(config_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        choices=[
            "binary_1",
            "binary_1_2",
            "binary_corr",
            "binary_2",
            "truthful_qa",
            "categorical",
        ],
        default="binary_1",
    )
    parser.add_argument(
        "--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--llm-layer", type=int, default=32, choices=range(0, 33)
    )
    # Llama-3 has 32 layers, CAA paper showed most effective steering around the
    # 13th layer for Llama-2 with 33 layers

    args = parser.parse_args()
    main(args)
