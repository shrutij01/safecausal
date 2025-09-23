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
import yaml

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_base(
    model, inputs, generate_config, interv_configs, **kwargs
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


def _intervene(model, hyperparameters, inputs, max_length) -> List[str]:
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
    model, inputs, generate_config, interv_configs
) -> List[str]:
    return _intervene(
        model,
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
        inputs,
        generate_config,
        interv_configs,
    )

    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated_text


def get_steering_vector(args, dataset_name="refusal"):
    """Load dataset and find the best steering vector dimension using correlation analysis."""
    # Load the specified dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", f"{dataset_name}.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{dataset_name.capitalize()} data not found at {data_path}"
        )

    dataset_data = data_utils.load_json(data_path)
    print(f"Loaded {len(dataset_data)} {dataset_name} samples")

    # Load SSAE model config to get the embedding layer
    config_path = os.path.join(args.modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Get the layer used for embeddings from the nested config structure
    embedding_layer = cfg.get("extra", {}).get(
        "llm_layer", 5
    )  # Default to 5 if not found
    print(f"Using embedding layer {embedding_layer} from SSAE config")

    # Create paired samples with proper labels like in data_utils.py
    questions = []
    labels = []

    # Set instruction based on dataset
    if dataset_name == "refusal":
        instruction = "For the following user request and model response, does the response refuse or comply with the user's request? Answer only with the option: '(A) Refuse' or '(B) Comply'.\n"
        concept_name = "refusal"
    elif dataset_name == "sycophancy":
        instruction = "For the following user request and model response, does the answer align with the user's behavior? Answer only with the option: '(A) Agree' or '(B) Disagree'.\n"
        concept_name = "sycophancy"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for item in dataset_data:
        question = item["question"]
        matching_answer = item["answer_matching_behavior"]  # Target behavior
        not_matching_answer = item[
            "answer_not_matching_behavior"
        ]  # Non-target behavior

        # Create paired samples: question + each answer choice
        matching_pair = f"{instruction}\n{question}\n{matching_answer}"
        not_matching_pair = f"{instruction}\n{question}\n{not_matching_answer}"

        # Add both samples with their labels
        # matching_answer should be labeled as 1 (target behavior)
        # not_matching_answer should be labeled as 0 (non-target behavior)
        questions.extend([not_matching_pair, matching_pair])
        labels.extend([0, 1])  # 0 for non-target, 1 for target behavior

    print(f"Created {len(questions)} question-answer pairs")
    if dataset_name == "refusal":
        print(
            f"Label distribution: {sum(labels)} refusals, {len(labels) - sum(labels)} compliances"
        )
    elif dataset_name == "sycophancy":
        print(
            f"Label distribution: {sum(labels)} sycophantic, {len(labels) - sum(labels)} non-sycophantic"
        )

    # Get embeddings for the questions using the same model and layer as will be used for steering
    print(f"Extracting embeddings from layer {embedding_layer}")
    embeddings = get_sentence_embeddings(
        questions,
        model_name="EleutherAI/pythia-70m-deduped",
        layer=embedding_layer,
    )
    print(f"Generated embeddings with shape: {np.array(embeddings).shape}")

    # Ensure embeddings are numpy array
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    print(f"Final embeddings shape: {embeddings.shape}")

    # Load SSAE model
    from ssae import DictLinearAE

    # Load model config
    config_path = os.path.join(args.modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (N, 1)
    print(f"Labels tensor shape: {labels_tensor.shape}")

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
        f"Best feature index: {best_feature_idx} with correlation: {max_correlation:.4f}"
    )
    print(f"This feature correlates with {concept_name} behavior")

    # Get the decoder column for this feature as steering vector
    decoder_weight = model.decoder.weight  # Shape: (rep_dim, hid_dim)
    steering_vector = decoder_weight[:, best_feature_idx]  # Shape: (rep_dim,)

    return (
        steering_vector,
        best_feature_idx,
        max_correlation,
        embedding_layer,
        dataset_name,
    )


def main(args, generate_configs):
    # Get steering vector using the specified dataset
    dataset_name = args.dataset
    steering_vector, feature_idx, correlation, used_layer, dataset_used = (
        get_steering_vector(args, dataset_name)
    )
    print(
        f"Using {dataset_name}-based steering vector from feature {feature_idx} (correlation: {correlation:.4f})"
    )
    print(
        f"Applying steering intervention to layer {used_layer} (same as embedding extraction)"
    )

    # Load dataset again for test prompts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", f"{dataset_name}.json")
    dataset_data = data_utils.load_json(data_path)

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
    else:
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

    # Get test prompts from dataset - extract just the user requests
    test_prompts = []
    for item in dataset_data[:10]:  # Use first 10 samples for testing
        # Extract just the question part (user request) from the dataset
        question = item["question"]
        # Remove the "Choices:" part and just keep the user request
        if "\n\nChoices:" in question:
            user_request = question.split("\n\nChoices:")[0]
        else:
            user_request = question
        test_prompts.append(user_request)

    print(
        f"Testing with {len(test_prompts)} user requests from {dataset_name} data"
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

    # Generate with positive steering (should increase target behavior)
    behavior_name = "refusal" if dataset_name == "refusal" else "sycophancy"
    opposite_name = (
        "compliance" if dataset_name == "refusal" else "non-sycophancy"
    )

    print(
        f"\n=== Generating with positive steering (increase {behavior_name}) ==="
    )
    positive_config = Box(
        {
            "type": "ssae",
            "max_length": args.max_length,
            "output_filename": f"positive_{dataset_name}_steering_generation.json",
        }
    )
    positive_text = generate(
        llm, tokenizer, inputs, positive_config, interv_configs
    )

    # Generate with negative steering (should decrease target behavior)
    print(
        f"\n=== Generating with negative steering (increase {opposite_name}) ==="
    )
    negative_interv_configs = []
    negative_interv_configs.append(
        Box(
            {
                "layer": used_layer,
                "scale": -args.steering_alpha,  # Negative scale to subtract the vector
                "steer_vec": steering_vector.to(device),
            }
        )
    )
    negative_config = Box(
        {
            "type": "ssae",
            "max_length": args.max_length,
            "output_filename": f"negative_{dataset_name}_steering_generation.json",
        }
    )
    negative_text = generate(
        llm, tokenizer, inputs, negative_config, negative_interv_configs
    )

    # Save results and print comparison
    results = {
        "dataset": dataset_name,
        "test_prompts": test_prompts,
        "baseline_responses": baseline_text,
        "positive_steering_responses": positive_text,
        "negative_steering_responses": negative_text,
        "steering_info": {
            "feature_idx": feature_idx,
            "correlation": correlation,
            "layer": used_layer,
            "steering_alpha": args.steering_alpha,
            "concept": behavior_name,
        },
    }

    save_path = os.path.join(
        BASE_DIR,
        "generated_outputs",
        f"{dataset_name}_steering_comparison.json",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Comparison Results ===")
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt[:100]}...")
        print(f"Baseline: {baseline_text[i][len(prompt):len(prompt)+150]}...")
        print(
            f"Positive Steering (+{behavior_name}): {positive_text[i][len(prompt):len(prompt)+150]}..."
        )
        print(
            f"Negative Steering (+{opposite_name}): {negative_text[i][len(prompt):len(prompt)+150]}..."
        )
        print("-" * 80)


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
        "--dataset",
        "-d",
        type=str,
        choices=["refusal", "sycophancy"],
        default="refusal",
        help="Dataset to use for steering vector extraction (default: refusal)",
    )
    parser.add_argument(
        "--llm", default="EleutherAI/pythia-70m-deduped", type=str
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
