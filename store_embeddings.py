import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import datetime
import os
import yaml

import utils

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_weighted_mean_embedding(tokens, model):
    with torch.no_grad():
        last_hidden_states = model(**tokens, output_hidden_states=True)[
            "hidden_states"
        ][-1]
    sequence_length = last_hidden_states.shape[1]
    weights = torch.linspace(1, sequence_length, steps=sequence_length)
    weights = weights / weights.sum()
    weights = weights.view(1, -1, 1)
    weighted_mean = (last_hidden_states * weights).sum(dim=1)
    return weighted_mean


def extract_embeddings(
    contexts,
    model,
    tokenizer,
):
    embeddings = []
    for context in tqdm(contexts):
        tokens = tokenizer(context, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            embeddings.append(get_weighted_mean_embedding(
                    tokens, model
                ).cpu().squeeze())
        del tokens
    return embeddings

def store_embeddings(filename, cfc1_train, cfc2_train, cfc1_test, cfc2_test):
    with h5py.File(filename, "w") as hdf_file:
        hdf_file.create_dataset(
            "cfc1_train", data=cfc1_train
        )
        hdf_file.create_dataset(
            "cfc2_train", data=cfc2_train
        )
        hdf_file.create_dataset(
            "cfc1_test", data=cfc1_test
        )
        hdf_file.create_dataset(
            "cfc2_test", data=cfc2_test
        )
    

def main(args):
    x_tuples = utils.load_dataset(
                dataset_name=args.dataset_name,
                task_type=args.task_type,
                d_type="str",
                n=args.dataset_length,
                string_length=args.string_length,
                cycle_distance=args.cycle_distance)
    split = int(0.9 * len(x_tuples))
    cfc1_train, cfc1_test = x_tuples[0: split, 0], x_tuples[split: , 0]
    cfc2_train, cfc2_test = x_tuples[0: split, 1], x_tuples[split: , 1]
    if args.dataset_name == "ana":
        instruction = "Swap the letters in this string."
    cfc1_train = utils.append_instruction(cfc1_train, instruction)
    cfc1_test = utils.append_instruction(cfc1_test, instruction)
    cfc2_train = utils.append_instruction(cfc2_train, instruction)
    cfc2_test = utils.append_instruction(cfc2_test, instruction)
    
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        args.model_id, token=ACCESS_TOKEN
    )
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_id,
        token=ACCESS_TOKEN,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,  # check compatibility
    )
    
    cfc1_train_embeddings = extract_embeddings(
        cfc1_train, model, tokenizer,
    )
    cfc1_test_embeddings = extract_embeddings(
        cfc1_test, model, tokenizer,
    )
    cfc2_train_embeddings = extract_embeddings(
        cfc2_train, model, tokenizer,
    )
    cfc2_test_embeddings = extract_embeddings(
        cfc2_test, model, tokenizer,
    )
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    directory_location = '/network/scratch/j/joshi.shruti/psp/'
    directory_name = os.path.join(directory_location, timestamp_str)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    embeddings_path = os.path.join(directory_name, "embeddings.h5")
    store_embeddings(
        embeddings_path,
        cfc1_train_embeddings,
        cfc2_train_embeddings,
        cfc1_test_embeddings,
        cfc2_test_embeddings,
    )
    
    config = {
        'dataset': args.dataset_name,
        'dataset_length': args.dataset_length,
        'task_type': args.task_type,
        'model': args.model_id,
    }
    config_path = os.path.join(directory_name, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_name", choices=["ana"]
    )
    parser.add_argument("model_id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--task-type", default="swap", choices=["swap", "cycle"]
    )
    parser.add_argument(
        "--dataset-length",
        default=10000,
    )
    parser.add_argument("--string-length", default=3)
    parser.add_argument("--cycle-distance", default=1)

    args = parser.parse_args()
    main(args)
    