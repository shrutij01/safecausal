import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import datetime
import os
import yaml

import psp.data_utils as utils

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_last_token_embedding(tokens, model, layer):
    with torch.no_grad():
        last_hidden_states = model(**tokens, output_hidden_states=True)[
            "hidden_states"
        ][layer][-1]
    return last_hidden_states.cpu()


def extract_embeddings(
    contexts,
    model,
    tokenizer,
):
    embeddings = []
    for context in tqdm(contexts):
        tokens = tokenizer(context, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings.append(
                get_last_token_embedding(tokens, model).cpu().squeeze()
            )
        del tokens
    return embeddings


def store_embeddings(filename, cfc1_train, cfc2_train, cfc1_test, cfc2_test):
    with h5py.File(filename, "w") as hdf_file:
        hdf_file.create_dataset("cfc1_train", data=cfc1_train)
        hdf_file.create_dataset("cfc2_train", data=cfc2_train)
        hdf_file.create_dataset("cfc1_test", data=cfc1_test)
        hdf_file.create_dataset("cfc2_test", data=cfc2_test)


def main(args):
    cfc_tuples, instruction = utils.load_dataset(
        dataset_name=args.dataset_name,
        file_path=args.gradeschooler_file_path,
    )
    split = int(0.9 * len(cfc_tuples))
    cfc_train, cfc_test = cfc_tuples[0:split], cfc_tuples[split:]
    import ipdb

    ipdb.set_trace()
    if args.append_instruction:
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
        # attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16,  # check compatibility
    )

    cfc1_train_embeddings = extract_embeddings(
        cfc1_train,
        model,
        tokenizer,
    )
    cfc1_test_embeddings = extract_embeddings(
        cfc1_test,
        model,
        tokenizer,
    )
    cfc2_train_embeddings = extract_embeddings(
        cfc2_train,
        model,
        tokenizer,
    )
    cfc2_test_embeddings = extract_embeddings(
        cfc2_test,
        model,
        tokenizer,
    )
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    directory_location = "/network/scratch/j/joshi.shruti/psp/"
    directory_name = os.path.join(
        directory_location, str(args.dataset_name), timestamp_str
    )
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
        "dataset": args.dataset_name,
        "dataset_length": args.dataset_length,
        "task_type": args.task_type,
        "model": args.model_id,
        "split": 0.9,
    }
    config_path = os.path.join(directory_name, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def store_eval_embeddings(args):
    embeddings_path = os.path.join(args.embedding_dir, "eval_embeddings_1.h5")
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
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
    cfc1_tuples, cfc2_tuples, _ = utils.load_dataset(
        dataset_name=args.dataset_name,
        task_type=args.task_type,
        d_type="str",
        n=args.dataset_length,
        string_length=args.string_length,
        cycle_distance=args.cycle_distance,
        file_path=args.gradeschooler_file_path,
    )
    cfc1_eval = extract_embeddings(cfc1_tuples, model, tokenizer)
    cfc2_eval = extract_embeddings(cfc2_tuples, model, tokenizer)
    with h5py.File(embeddings_path, "w") as hdf_file:
        hdf_file.create_dataset("cfc1_eval", data=cfc1_eval)
        hdf_file.create_dataset("cfc2_eval", data=cfc2_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-file-path",
    )
    parser.add_argument("--eval", default=False)
    parser.add_argument(
        "--dataset_name",
        choices=["binary_1", "truthful_qa"],
        default="binary_1",
    )
    parser.add_argument(
        "--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--dataset-length",
        default=10000,
    )
    parser.add_argument(
        "--embedding_dir",
        default="/network/scratch/j/joshi.shruti/psp/gradeschooler/2024-08-22_23-37-25",
    )
    parser.add_argument("--append-instruction", default=False)

    args = parser.parse_args()
    print(args.eval)
    if args.eval:
        import ipdb

        ipdb.set_trace()
        store_eval_embeddings(args)
    else:
        main(args)

# export PYTHONPATH=${PYTHONPATH}:/home/mila/j/joshi.shruti/causalrepl_space/psp
