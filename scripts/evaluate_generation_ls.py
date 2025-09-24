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
import utils.data_utils as data_utils
import numpy as np
from scripts.evaluate_labeled_sentences import get_sentence_embeddings
from transformers import AutoModelForCausalLM
import yaml

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_base(
    model, tokenizer, inputs, generate_config, interv_configs, **kwargs
) -> List[str]:
    """
    Generate text without any interventions.
    """
    output = model.generate(
        **inputs,
        max_length=generate_config.max_length,
        do_sample=True,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
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
            _output = (_output + scale * steer_vec.to(_output.dtype)).to(
                _output.dtype
            )
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

        # Support multiple model architectures
        if hasattr(model, "gpt_neox"):  # Pythia models
            target_layer = model.gpt_neox.layers[layer]
        elif hasattr(model, "model") and hasattr(
            model.model, "layers"
        ):  # Llama and Gemma models
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
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
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

    # Get input length to slice off the prompt from generated output
    input_length = inputs["input_ids"].shape[1]

    # Slice to get only newly generated tokens (excluding the input prompt)
    generated_tokens = output[:, input_length:]

    generated_text = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return generated_text


def get_steering_vectors_for_features(args, features_list):
    """Load labeled-sentences dataset and find steering vectors for specified features using correlation analysis."""
    # Load labeled sentences data
    from scripts.evaluate_labeled_sentences import load_labeled_sentences_test

    sentences, all_labels = load_labeled_sentences_test()
    print(f"Loaded {len(sentences)} labeled sentences with {len(all_labels)} label types")

    # Load SSAE model config to get the embedding layer
    config_path = os.path.join(args.modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Get the layer used for embeddings from the nested config structure
    embedding_layer = cfg.get("llm_layer", cfg.get("extra", {}).get("llm_layer", 5))
    print(f"Using embedding layer {embedding_layer} from SSAE config")

    # Get embeddings for the sentences using the same model and layer as will be used for steering
    print(f"Extracting embeddings from layer {embedding_layer}")
    embeddings = get_sentence_embeddings(
        sentences,
        model_name=args.llm,
        layer=embedding_layer,
    )
    print(f"Generated embeddings with shape: {np.array(embeddings).shape}")

    # Ensure embeddings are numpy array
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    print(f"Final embeddings shape: {embeddings.shape}")

    # Load SSAE model
    from ssae import DictLinearAE

    # Load model weights
    weights_path = os.path.join(args.modeldir, "weights.pth")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Get dimensions from the saved weights
    rep_dim = state_dict["encoder.weight"].shape[1]  # input dimension
    hid_dim = state_dict["encoder.weight"].shape[0]  # hidden dimension
    print(f"Model dimensions: rep_dim={rep_dim}, hid_dim={hid_dim}")

    # Create and load model
    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    # Get SSAE activations
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        acts = model.encoder(
            embeddings_tensor
        )  # Shape: (N, F) where F is number of features
    print(f"SSAE activations shape: {acts.shape}")

    # Get decoder weight for extracting steering vectors
    decoder_weight = model.decoder.weight  # Shape: (rep_dim, hid_dim)

    steering_vectors = {}

    # Process each requested feature
    for feature_type, feature_value in features_list:
        feature_key = f"{feature_type}-{feature_value}"

        if feature_key not in all_labels:
            print(f"Warning: Feature '{feature_key}' not found in labels. Available: {list(all_labels.keys())}")
            continue

        # Convert labels to tensor (binarize)
        labels = all_labels[feature_key]
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)
        print(f"Processing feature '{feature_key}' with {sum(labels)}/{len(labels)} positive labels")

        # Compute correlation using the same method as evaluate_labeled_sentences.py
        acts_centered = acts - acts.mean(dim=0, keepdim=True)
        acts_std = acts_centered.norm(dim=0, keepdim=True)

        label_centered = labels_tensor - labels_tensor.mean(dim=0, keepdim=True)
        label_std = label_centered.norm(dim=0, keepdim=True)

        # Correlation computation
        numerator = acts_centered.T @ label_centered  # F × 1
        denominator = acts_std.T * label_std  # F × 1

        mask = denominator != 0
        corr_vector = torch.zeros_like(numerator)
        corr_vector[mask] = numerator[mask] / denominator[mask]

        # Find the feature with maximum absolute correlation
        best_feature_idx = corr_vector.abs().argmax().item()
        max_correlation = corr_vector[best_feature_idx].item()

        print(
            f"Feature '{feature_key}': best dimension {best_feature_idx} with correlation {max_correlation:.4f}"
        )

        # Get the decoder column for this feature as steering vector
        steering_vector = decoder_weight[:, best_feature_idx]  # Shape: (rep_dim,)

        steering_vectors[feature_key] = {
            'vector': steering_vector,
            'feature_idx': best_feature_idx,
            'correlation': max_correlation,
            'layer': embedding_layer
        }

    return steering_vectors


def main(args, generate_configs):
    # Define features to extract steering vectors for
    features_list = [
        ("domain", "fantasy"),
        ("domain", "science"),
        ("domain", "news"),
        ("domain", "other"),
        ("sentiment", "positive"),
        ("sentiment", "neutral"),
        ("sentiment", "negative"),
        ("voice", "active"),
        ("voice", "passive"),
    ]

    # Get steering vectors for all requested features
    steering_vectors = get_steering_vectors_for_features(args, features_list)

    if not steering_vectors:
        print("No valid steering vectors found!")
        return

    # Use the feature specified by user or default to first available
    if hasattr(args, 'feature') and args.feature:
        selected_feature = args.feature
        if selected_feature not in steering_vectors:
            print(f"Feature '{selected_feature}' not available. Available: {list(steering_vectors.keys())}")
            return
    else:
        selected_feature = list(steering_vectors.keys())[0]
        print(f"No specific feature requested, using first available: {selected_feature}")

    steering_info = steering_vectors[selected_feature]
    steering_vector = steering_info['vector']
    feature_idx = steering_info['feature_idx']
    correlation = steering_info['correlation']
    used_layer = steering_info['layer']

    print(
        f"Using feature '{selected_feature}' steering vector from dimension {feature_idx} (correlation: {correlation:.4f})"
    )
    print(
        f"Applying steering intervention to layer {used_layer} (same as embedding extraction)"
    )

    # Use predefined test prompts for generation
    test_prompts = [
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

    interv_configs = []
    interv_configs.append(
        Box(
            {
                "layer": used_layer,  # Use the same layer as embedding extraction
                "scale": args.steering_alpha,
                "steer_vec": steering_vector.to(device),
            }
        )
    )
    # Load the appropriate model based on args.llm
    if "pythia" in args.llm.lower():
        from transformers import GPTNeoXForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.llm)
        llm = GPTNeoXForCausalLM.from_pretrained(
            args.llm, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
    elif "gemma" in args.llm.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.llm)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
    else:
        # Default to Llama-style loading for other models
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            args.llm, token=ACCESS_TOKEN
        )
        llm = transformers.LlamaForCausalLM.from_pretrained(
            args.llm,
            token=ACCESS_TOKEN,
            low_cpu_mem_usage=True,
            cache_dir="./model_cache",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(
        f"Testing with {len(test_prompts)} predefined prompts"
    )
    inputs = tokenizer(
        test_prompts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = inputs.to(device)

    # Generate baseline (no intervention)
    print("\n=== Generating baseline responses (no steering) ===")
    baseline_config = Box(
        {
            "type": "base",
            "max_length": args.max_length,
            "output_filename": "baseline_generation.json",
        }
    )
    baseline_text = generate(llm, tokenizer, inputs, baseline_config, [])

    # Generate with positive steering (should increase target feature)
    behavior_name = selected_feature.replace("-", " ")

    print(
        f"\n=== Generating with positive steering (enhance {behavior_name}) ==="
    )
    positive_config = Box(
        {
            "type": "ssae",
            "max_length": args.max_length,
            "output_filename": f"positive_{selected_feature}_steering_generation.json",
        }
    )
    positive_text = generate(
        llm, tokenizer, inputs, positive_config, interv_configs
    )


    # Generate with random steering vector
    print(f"\n=== Generating with random steering vector ===")
    random_vector = torch.randn_like(steering_vector).to(device)
    random_vector = random_vector / random_vector.norm()  # Normalize
    random_interv_configs = []
    random_interv_configs.append(
        Box(
            {
                "layer": used_layer,
                "scale": args.steering_alpha,
                "steer_vec": random_vector,
            }
        )
    )
    random_config = Box(
        {
            "type": "ssae",
            "max_length": args.max_length,
            "output_filename": f"random_steering_generation.json",
        }
    )
    random_text = generate(
        llm, tokenizer, inputs, random_config, random_interv_configs
    )

    # Restructure results prompt by prompt for all prompts
    prompt_results = []
    for i in range(len(test_prompts)):
        prompt_result = {
            "prompt_id": i + 1,
            "original_prompt": test_prompts[i],
            "baseline_generation": baseline_text[i],
            "positive_steering_generation": positive_text[i],
            "random_steering_generation": random_text[i],
        }
        prompt_results.append(prompt_result)

    # Save results
    results = {
        "dataset": "labeled-sentences",
        "selected_feature": selected_feature,
        "steering_info": {
            "feature_idx": feature_idx,
            "correlation": correlation,
            "layer": used_layer,
            "steering_alpha": args.steering_alpha,
            "concept": behavior_name,
        },
        "all_steering_vectors": {k: {"feature_idx": v["feature_idx"], "correlation": v["correlation"]} for k, v in steering_vectors.items()},
        "prompt_results": prompt_results,
    }

    save_path = os.path.join(
        BASE_DIR,
        "generated_outputs",
        f"labeled_sentences_{selected_feature}_steering_comparison.json",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Comparison Results ===")
    for i in range(len(test_prompts)):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}: {test_prompts[i]}")
        print(f"{'='*80}")

        print(f"\nBASELINE:")
        print(f"{baseline_text[i]}")

        print(f"\nPOSITIVE STEERING (enhance {behavior_name}):")
        print(f"{positive_text[i]}")

        print(f"\nRANDOM STEERING:")
        print(f"{random_text[i]}")

        print(f"\n{'-'*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datafile", help="Optional datafile (not used when using steering)"
    )
    parser.add_argument(
        "--dataconfig",
        help="Optional dataconfig (not used when using steering)",
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        help="Specific feature to use for steering (e.g., 'domain-fantasy', 'sentiment-positive'). If not specified, uses first available.",
    )
    parser.add_argument(
        "--llm",
        default="EleutherAI/pythia-70m-deduped",
        type=str,
        help="LLM model to use for generation (e.g., EleutherAI/pythia-70m-deduped, google/gemma-2-2b-it, meta-llama/Meta-Llama-3.1-8B-Instruct)",
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
    main(args, None)  # We don't use generate_configs anymore
