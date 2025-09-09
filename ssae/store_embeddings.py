from h5py._hl import dataset
import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import os
import yaml

import utils.data_utils as utils

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
    cfc_train_tuples, cfc_test_tuples = utils.load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
    )
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
        cfc_train_tuples,
        model,
        tokenizer,
        args.llm_layer,
    )
    cfc_test_embeddings = extract_embeddings(
        cfc_test_tuples,
        model,
        tokenizer,
        args.llm_layer,
    )
    directory_location = "/network/scratch/j/joshi.shruti/ssae/"
    directory_name = os.path.join(directory_location, str(args.dataset))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if args.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model_name = "llama3"
    elif args.model_id == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        model_name = "r1llama3"
    else:
        raise NotImplementedError
    embeddings_path = os.path.join(
        directory_name,
        "L_"
        + str(args.llm_layer)
        + "_M_"
        + str(model_name)
        + str(args.dataset)
        + ".h5",
    )
    store_embeddings(
        embeddings_path,
        cfc_train_embeddings,
        cfc_test_embeddings,
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
        num_concepts = 2
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
        "llm_layer": args.llm_layer,
        "split": 0.9,
    }
    config_path = os.path.join(
        directory_name,
        "L_" + str(args.llm_layer) + str(args.dataset) + ".yaml",
    )
    with open(config_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "eng-french",
            "eng-german",
            "masc-fem-eng",
            "masc-fem-mixed",
            "2-binary",
            "corr-binary",
            "truthful-qa",
            "categorical",
            "safearena",
            "wildjailbreak",
            "labeled-sentences",
        ],
        default="eng-french",
    )
    parser.add_argument(
        "--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--llm-layer", type=int, default=32, choices=range(0, 33)
    )
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
    # Llama-3 has 32 layers, CAA paper showed most effective steering around the
    # 13th layer for Llama-2 with 33 layers

    args = parser.parse_args()
    main(args)
