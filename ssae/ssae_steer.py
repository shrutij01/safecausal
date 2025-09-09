from typing import List
import argparse
from collections import Counter
import torch
import torch.nn.functional as F
import yaml
from re import L
import data_utils as utils
import os

from ssae import DictLinearAE

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml
from box import Box
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_base(
    model, tokenizer, inputs, generate_config, **kwargs
) -> List[str]:
    """
    Generate text without any interventions.
    """
    output = model.generate(
        **inputs,
        max_length=generate_config.max_length,
        do_sample=True,
    )
    return output


def _intervene(
    model, tokenizer, hyperparameters, inputs, max_length
) -> List[str]:
    # Set up hook for forward pass to inject probe vector.
    def hook_model(steer_vec, scale):
        def forward_hook(module, input, output):
            _output = output
            if isinstance(output, tuple):
                _output = output[0]
            # orig_norm = _output.norm()
            _output = (_output + scale * steer_vec).float()
            # _output = _output * (orig_norm / _output.norm())
            return (
                (_output, *output[1:])
                if isinstance(output, tuple)
                else _output
            )

        return forward_hook

    hooks = []
    for h_param in hyperparameters:
        layer = h_param.layer
        scale = float(h_param.scale)
        steer_vec = h_param.steer_vec
        steer_vec = steer_vec / steer_vec.norm()
        hook_func = hook_model(steer_vec, scale)

        # Support both GPT-NeoX (Pythia) and Llama architectures
        if hasattr(model, "gpt_neox"):  # Pythia models
            target_layer = model.gpt_neox.layers[layer]
        elif hasattr(model, "model"):  # Llama models
            target_layer = model.model.layers[layer]
        else:
            raise ValueError(
                f"Unsupported model architecture. Model has attributes: {dir(model)}"
            )

        hook = target_layer.register_forward_hook(hook_func)
        hooks.append(hook)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
    )
    for hook in hooks:
        hook.remove()
    return outputs


def _generate_ssae(
    model, tokenizer, inputs, generate_config, interv_configs
) -> List[str]:
    return _intervene(
        model,
        tokenizer,
        interv_configs,
        inputs,
        generate_config.max_length,
    )


def generate(
    model, tokenizer, inputs, generate_config, interv_configs
) -> List[str]:
    generate_func = {
        "base": _generate_base,
        "ssae": _generate_ssae,
    }[generate_config.type]
    output = generate_func(
        model,
        tokenizer,
        inputs,
        generate_config,
        interv_configs,
    )

    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated_text


def get_most_frequent_index(indices):
    if indices is None:
        print("The list is empty.")
        return None

    indices_list = [int(i.item()) for i in indices]
    counter = Counter(indices_list)
    most_frequent_index, count = counter.most_common(1)[0]
    print(
        f"The most common index is {most_frequent_index} with count {count} / {len(indices_list)}."
    )
    return most_frequent_index


def get_max_cos_and_steering_vector_for_concept(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
):
    """
    z: [B, D] tensor of original vectors
    decoder_weight: [V, D] decoder weight matrix (rows = token embeddings)

    Returns:
        mean and std of max cosine similarities over decoder directions
    """
    z = F.normalize(z, dim=1)  # [B, D]
    z_tilde = F.normalize(z_tilde, dim=1)  # [B, D]
    # decoder = F.normalize(
    #     decoder_weight, dim=1
    # )  # [V, D] — columns as directions
    decoder = decoder_weight.to(z.device)
    decoder_bias = F.normalize(decoder_bias, dim=0)
    decoder_bias = decoder_bias.to(z.device)
    B, D = z.shape
    V = decoder.shape[0]

    # z: [B, D], decoder: [D, V]
    z_tilde_hat = (
        z.unsqueeze(2)
        + decoder.unsqueeze(0)
        + (decoder_bias.unsqueeze(0)).unsqueeze(2)
    )  # [B, D, V]
    z_tilde_hat = F.normalize(
        z_tilde_hat, dim=1
    )  # normalize shifted vectors: [B, D, V]

    z_tilde = z_tilde.unsqueeze(2)  # [B, D, 1]
    cosines = torch.bmm(z_tilde.transpose(1, 2), z_tilde_hat).squeeze(
        1
    )  # [B, V]
    max_cosines = cosines.max(dim=1).values  # [B]
    indices = cosines.argmax(dim=1)  # [B]
    most_frequent_index = get_most_frequent_index(indices=indices)
    steering_vector = (
        decoder[:, most_frequent_index]
        if most_frequent_index is not None
        else None
    )
    return max_cosines.mean().item(), max_cosines.std().item(), steering_vector


def load_model_config(modeldir: str) -> Box:
    config_path = os.path.join(modeldir, "model_config.yaml")
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config


def load_ssae(
    modeldir: str, dataconfig: Box
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a DictLinearAE model and return the decoder weights as a numpy array.
    """
    modelconfig = load_model_config(modeldir)
    model = DictLinearAE(
        rep_dim=dataconfig.rep_dim,
        hid=int(
            modelconfig.num_concepts * modelconfig.overcompleteness_factor
        ),
        norm_type=modelconfig.norm_type,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(modeldir, "sparse_dict_model.pth"))
    )
    model.eval()
    return (
        model.decoder.weight.data,
        model.decoder.bias.data,
        model.encoder.weight.data,
        model.encoder.bias.data,
    )


def main(args, generate_configs):
    tilde_z_test, z_test = utils.load_test_data(
        datafile=args.datafile,
    )
    z_test = utils.tensorify(z_test, device)
    tilde_z_test = utils.tensorify(tilde_z_test, device)
    with open(args.dataconfig, "r") as file:
        dataconfig = Box(yaml.safe_load(file))
    modeldirs = args.modeldirs
    print("Loading decoder weight matrices...")
    decoder_weight_matrices = [
        load_ssae(modeldir, dataconfig)[0] for modeldir in modeldirs
    ]
    decoder_bias_vectors = [
        load_ssae(modeldir, dataconfig)[1] for modeldir in modeldirs
    ]
    mean_cos, std_cos, steering_vector = (
        get_max_cos_and_steering_vector_for_concept(
            z_test,
            tilde_z_test,
            decoder_weight_matrices[0],
            decoder_bias_vectors[0],
        )
    )
    interv_configs = []
    interv_configs.append(
        Box(
            {
                "layer": args.steering_layer,
                "scale": args.steering_alpha,
                "steer_vec": steering_vector.to(device),
            }
        )
    )
    llm = AutoModelForCausalLM.from_pretrained(args.llm).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = [
        "The",
        "When",
        "Why",
        "How",
        "What",
        "Where",
        "This",
        "That",
        "If",
        "But",
        "So",
        "With",
        "Because",
        "Although",
        "However",
        "Therefore",
        "Furthermore",
        "Meanwhile",
        "Thus",
        "Who",
        "Yet",
        "Unless",
        "As",
        "I",
        "You",
        "He",
        "She",
        "They",
        "We",
        "It",
        "My",
        "Your",
        "His",
        "Her",
        "Their",
        "Our",
        "Mine",
        "Yours",
        "Hers",
        "Theirs",
        "Ours",
    ]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = inputs.to(device)
    for gen_config in generate_configs:
        generated_text = generate(
            llm, tokenizer, inputs, gen_config, interv_configs
        )
        save_path = os.path.join(
            BASE_DIR, "generated_outputs", gen_config.output_filename
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(generated_text, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile")
    parser.add_argument("dataconfig")
    parser.add_argument(
        "--modeltype", default="ssae", choices=["llamascope", "ssae"]
    )
    parser.add_argument(
        "--llm", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str
    )
    parser.add_argument(
        "--modeldirs",
        nargs="+",
        type=str,
        help="List of model directories to compare.",
    )
    parser.add_argument(
        "--steering-layer",
        "-sl",
        type=int,
        default=16,
        help="Starting layer index to apply steering intervention (applies to this layer and all following layers, default: 16)",
    )
    parser.add_argument(
        "--steering-alpha",
        "-sa",
        type=float,
        default=5.0,
        help="Steering strength multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--max-length",
        "-ml",
        type=int,
        default=50,
        help="Maximum length of the generated text (default: 50)",
    )
    args = parser.parse_args()
    generate_configs = [
        {
            "type": "base",
            "output_filename": "base_generation.json",
        },
        {
            "type": "ssae",
            "output_filename": "ssae_generation.json",
            "steering_configs": [
                {
                    "feature_key": "feminine",
                    "feature_value": "positive",
                },
            ],
        },
    ]
    main(args, generate_configs)
