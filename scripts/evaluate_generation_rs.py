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

try:
    from loaders.modelloader import load_gemmascope_checkpoint, load_ssae_models
except ImportError:
    # Fallback for direct import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from loaders.modelloader import load_gemmascope_checkpoint, load_ssae_models
import utils.data_utils as data_utils
import numpy as np
from scripts.evaluate_labeled_sentences import get_sentence_embeddings
from transformers import AutoModelForCausalLM
import yaml

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GemmaScopeModel:
    """Wrapper for Gemma Scope model to match SSAE interface."""
    def __init__(self, decoder_weight, decoder_bias, encoder_weight, encoder_bias):
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.encoder_weight = encoder_weight
        self.encoder_bias = encoder_bias

    def encoder(self, x):
        """Apply encoder transformation."""
        return torch.nn.functional.relu(torch.matmul(x, self.encoder_weight.T) + self.encoder_bias)

    def decoder(self, x):
        """Apply decoder transformation."""
        return torch.matmul(x, self.decoder_weight.T) + self.decoder_bias

    def forward(self, x):
        """Full forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def eval(self):
        """Set to eval mode (no-op for this implementation)."""
        pass


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


def compute_mcc_comparison(args, dataset_name="refusal"):
    """Compare MCC values between SSAE and Gemma Scope models."""
    # Load the specified dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", f"{dataset_name}.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{dataset_name.capitalize()} data not found at {data_path}"
        )

    dataset_data = data_utils.load_json(data_path)
    print(f"Loaded {len(dataset_data)} {dataset_name} samples")

    # Limit test samples for faster processing
    max_samples = getattr(args, 'max_test_samples', 100)
    if len(dataset_data) > max_samples:
        dataset_data = dataset_data[:max_samples]
        print(f"Limited to {max_samples} samples for faster processing")

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
    # Use the same model that will be used for generation to ensure consistency
    embeddings = get_sentence_embeddings(
        questions,
        model_name=args.llm,
        layer=embedding_layer,
    )
    print(f"Generated embeddings with shape: {np.array(embeddings).shape}")

    # Ensure embeddings are numpy array
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    print(f"Final embeddings shape: {embeddings.shape}")
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (N, 1)
    print(f"Labels tensor shape: {labels_tensor.shape}")

    # 1. Load and evaluate SSAE model
    print(f"\n{'='*60}")
    print("EVALUATING SSAE MODEL")
    print(f"{'='*60}")

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
    print(f"SSAE dimensions: rep_dim={rep_dim}, hid_dim={hid_dim}")

    # Create and load model
    ssae_model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    ssae_model.load_state_dict(state_dict)
    ssae_model.eval()

    # Get SSAE activations
    with torch.no_grad():
        ssae_acts = ssae_model.encoder(embeddings_tensor)  # Shape: (N, F)
    print(f"SSAE activations shape: {ssae_acts.shape}")

    # Compute SSAE correlations
    ssae_corr_vector = compute_correlations(ssae_acts, labels_tensor)
    ssae_best_idx = ssae_corr_vector.abs().argmax().item()
    ssae_max_corr = ssae_corr_vector[ssae_best_idx].item()

    print(f"SSAE - Best feature: {ssae_best_idx}, MCC: {ssae_max_corr:.4f}")

    # 2. Load and evaluate Gemma Scope model
    print(f"\n{'='*60}")
    print("EVALUATING GEMMA SCOPE MODEL")
    print(f"{'='*60}")

    # Load Gemma Scope checkpoint
    gemma_decoder_weight, gemma_decoder_bias, gemma_encoder_weight, gemma_encoder_bias = load_gemmascope_checkpoint()

    print(f"Gemma Scope dimensions: encoder_weight={gemma_encoder_weight.shape}, decoder_weight={gemma_decoder_weight.shape}")

    # Create Gemma Scope model
    gemma_model = GemmaScopeModel(gemma_decoder_weight, gemma_decoder_bias, gemma_encoder_weight, gemma_encoder_bias)

    # Get Gemma Scope activations
    with torch.no_grad():
        gemma_acts = gemma_model.encoder(embeddings_tensor)  # Shape: (N, F)
    print(f"Gemma Scope activations shape: {gemma_acts.shape}")

    # Compute Gemma Scope correlations
    gemma_corr_vector = compute_correlations(gemma_acts, labels_tensor)
    gemma_best_idx = gemma_corr_vector.abs().argmax().item()
    gemma_max_corr = gemma_corr_vector[gemma_best_idx].item()

    print(f"Gemma Scope - Best feature: {gemma_best_idx}, MCC: {gemma_max_corr:.4f}")

    # 3. Comparison results
    print(f"\n{'='*60}")
    print("MCC COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Concept: {concept_name}")
    print(f"Embedding layer: {embedding_layer}")
    print(f"Number of samples: {len(labels)}")
    print(f"\nSSAE Model:")
    print(f"  Best feature index: {ssae_best_idx}")
    print(f"  MCC value: {ssae_max_corr:.4f}")
    print(f"  Model dimensions: {rep_dim} -> {hid_dim}")
    print(f"\nGemma Scope Model:")
    print(f"  Best feature index: {gemma_best_idx}")
    print(f"  MCC value: {gemma_max_corr:.4f}")
    print(f"  Model dimensions: {gemma_encoder_weight.shape[1]} -> {gemma_encoder_weight.shape[0]}")
    print(f"\nComparison:")
    mcc_diff = gemma_max_corr - ssae_max_corr
    print(f"  MCC difference (Gemma - SSAE): {mcc_diff:.4f}")
    if abs(mcc_diff) < 0.01:
        print(f"  Result: Similar performance")
    elif mcc_diff > 0:
        print(f"  Result: Gemma Scope performs better (+{mcc_diff:.4f})")
    else:
        print(f"  Result: SSAE performs better (+{abs(mcc_diff):.4f})")

    # Return comparison results
    return {
        "dataset": dataset_name,
        "concept": concept_name,
        "embedding_layer": embedding_layer,
        "num_samples": len(labels),
        "ssae": {
            "best_feature_idx": ssae_best_idx,
            "mcc": ssae_max_corr,
            "dimensions": (rep_dim, hid_dim)
        },
        "gemma_scope": {
            "best_feature_idx": gemma_best_idx,
            "mcc": gemma_max_corr,
            "dimensions": (gemma_encoder_weight.shape[1], gemma_encoder_weight.shape[0])
        },
        "comparison": {
            "mcc_difference": mcc_diff,
            "winner": "gemma_scope" if mcc_diff > 0.01 else "ssae" if mcc_diff < -0.01 else "similar"
        }
    }


def compute_correlations(activations, labels):
    """Compute correlations between activations and labels."""
    # Center the data
    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)

    label_centered = labels - labels.mean(dim=0, keepdim=True)
    label_std = label_centered.norm(dim=0, keepdim=True)

    # Correlation computation
    numerator = acts_centered.T @ label_centered  # F × 1
    denominator = acts_std.T * label_std  # F × 1

    mask = denominator != 0
    corr_vector = torch.zeros_like(numerator)
    corr_vector[mask] = numerator[mask] / denominator[mask]

    return corr_vector.squeeze()


def get_steering_vector(args, dataset_name="refusal"):
    """Load dataset and find the best steering vector dimension using correlation analysis.
    This function maintains backward compatibility for generation functionality."""

    # Use the new comparison function but return only SSAE results for backward compatibility
    comparison_results = compute_mcc_comparison(args, dataset_name)

    # Extract SSAE-specific results for steering
    ssae_results = comparison_results["ssae"]

    # Load SSAE model for steering vector extraction
    from ssae import DictLinearAE
    config_path = os.path.join(args.modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weights_path = os.path.join(args.modeldir, "weights.pth")
    state_dict = torch.load(weights_path, map_location="cpu")

    rep_dim = state_dict["encoder.weight"].shape[1]
    hid_dim = state_dict["encoder.weight"].shape[0]

    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    # Get the decoder column for the best feature as steering vector
    decoder_weight = model.decoder.weight  # Shape: (rep_dim, hid_dim)
    steering_vector = decoder_weight[:, ssae_results["best_feature_idx"]]  # Shape: (rep_dim,)

    return (
        steering_vector,
        ssae_results["best_feature_idx"],
        ssae_results["mcc"],
        comparison_results["embedding_layer"],
        dataset_name,
    )


def main(args, generate_configs):
    dataset_name = args.dataset

    # Check if MCC-only mode is enabled
    if args.mcc_only:
        print("=" * 80)
        print("MCC COMPARISON MODE - SSAE vs GEMMA SCOPE")
        print("=" * 80)

        # Run MCC comparison only
        comparison_results = compute_mcc_comparison(args, dataset_name)

        # Save results if output file specified
        if args.output_json:
            import json
            with open(args.output_json, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\nResults saved to {args.output_json}")

        return comparison_results

    # Original generation functionality
    # Get steering vector using the specified dataset
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

    # Restructure results prompt by prompt for first 10 prompts
    prompt_results = []
    for i in range(min(10, len(test_prompts))):
        prompt_result = {
            "prompt_id": i + 1,
            "original_prompt": test_prompts[i],
            "baseline_generation": baseline_text[i],
            "positive_steering_generation": positive_text[i],
            "negative_steering_generation": negative_text[i],
            "random_steering_generation": random_text[i],
        }
        prompt_results.append(prompt_result)

    # Save results
    results = {
        "dataset": dataset_name,
        "steering_info": {
            "feature_idx": feature_idx,
            "correlation": correlation,
            "layer": used_layer,
            "steering_alpha": args.steering_alpha,
            "concept": behavior_name,
        },
        "prompt_results": prompt_results,
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
    for i in range(min(10, len(test_prompts))):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}: {test_prompts[i]}")
        print(f"{'='*80}")

        print(f"\nBASELINE:")
        print(f"{baseline_text[i]}")

        print(f"\nPOSITIVE STEERING (+{behavior_name}):")
        print(f"{positive_text[i]}")

        print(f"\nNEGATIVE STEERING (+{opposite_name}):")
        print(f"{negative_text[i]}")

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
        "--dataset",
        "-d",
        type=str,
        choices=["refusal", "sycophancy"],
        default="refusal",
        help="Dataset to use for steering vector extraction (default: refusal)",
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
    parser.add_argument(
        "--mcc-only",
        action="store_true",
        help="Only compute MCC comparison between SSAE and Gemma Scope, skip generation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save MCC comparison results to JSON file",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=100,
        help="Maximum number of test samples to process (default: 100)",
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
