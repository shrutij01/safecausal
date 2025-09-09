from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
import transformers
from box import Box
import json

from loaders import TestDataLoader, load_ssae_models

ACCESS_TOKEN = "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"
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


def main(args, generate_configs):
    loader = TestDataLoader(device=device, verbose=args.verbose)
    tensors, dataconfig, test_labels, status = loader.load_test_data(
        args.datafile, args.dataconfig
    )

    if tensors is None:
        print(f"Failed to load test data: {status}")
        return

    tilde_z_test, z_test = tensors
    print(f"Data loaded successfully: {status}")

    # Split data by concept to get steering vector
    concept_test_sets = loader.split_by_label(
        tilde_z=tilde_z_test, z=z_test, labels=test_labels
    )

    if concept_test_sets is None:
        print("Failed to split data by concepts")
        return

    # Load SSAE model (only need one for steering vector)
    decoder_weight_matrices, decoder_bias_vectors, _, _ = load_ssae_models(
        [args.modeldir]
    )
    decoder_weight = decoder_weight_matrices[0]
    decoder_bias = decoder_bias_vectors[0]

    # Get steering vector from first concept (you may want to specify which concept)
    first_concept = list(concept_test_sets.keys())[0]
    concept_tilde_z, concept_z = concept_test_sets[first_concept]

    # Import the cosine similarity function from evaluate_cosinesim
    from scripts.evaluate_cosinesim import (
        get_max_cos_and_steering_vector_for_concept,
    )

    _, _, steering_vector = get_max_cos_and_steering_vector_for_concept(
        concept_z, concept_tilde_z, decoder_weight, decoder_bias
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
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        args.llm, token=ACCESS_TOKEN
    )
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.llm,
        token=ACCESS_TOKEN,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="./model_cache",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,  # check compatibility
    ).to(device)
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
        "--llm", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=True,
        help="SSAE model directory for steering vector extraction.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
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
        Box(
            {
                "type": "base",
                "max_length": args.max_length,
                "output_filename": "base_generation.json",
            }
        ),
        Box(
            {
                "type": "ssae",
                "max_length": args.max_length,
                "output_filename": "ssae_generation.json",
            }
        ),
    ]
    main(args, generate_configs)
