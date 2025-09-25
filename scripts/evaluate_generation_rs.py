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
    from loaders.modelloader import (
        load_gemmascope_checkpoint,
        load_pythia_sae_checkpoint,
        load_ssae_models,
    )
except ImportError:
    # Fallback for direct import
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from loaders.modelloader import (
        load_gemmascope_checkpoint,
        load_pythia_sae_checkpoint,
        load_ssae_models,
    )

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import utils.data_utils as data_utils
import numpy as np
from scripts.evaluate_labeled_sentences import get_sentence_embeddings
from transformers import AutoModelForCausalLM
import yaml

ACCESS_TOKEN = "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"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ExternalSAEModel:
    """Wrapper for external SAE models (Gemma Scope, Pythia SAE) to match SSAE interface."""

    def __init__(
        self,
        decoder_weight,
        decoder_bias,
        encoder_weight,
        encoder_bias,
        model_name="External SAE",
    ):
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.encoder_weight = encoder_weight
        self.encoder_bias = encoder_bias
        self.model_name = model_name

    def encoder(self, x):
        """Apply encoder transformation."""
        return torch.nn.functional.relu(
            torch.matmul(x, self.encoder_weight.T) + self.encoder_bias
        )

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
    """Simple MCC comparison between SSAE and appropriate external SAE model."""
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", f"{dataset_name}.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{dataset_name.capitalize()} data not found at {data_path}"
        )

    dataset_data = data_utils.load_json(data_path)

    # Limit to 100 samples
    max_samples = getattr(args, "max_test_samples", 100)
    if len(dataset_data) > max_samples:
        dataset_data = dataset_data[:max_samples]

    # Get embedding layer from SSAE config
    config_path = os.path.join(args.modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    embedding_layer = cfg.get("extra", {}).get("llm_layer", 5)

    # Create question-answer pairs with labels
    questions = []
    labels = []

    instruction = (
        "For the following user request and model response, does the response refuse or comply with the user's request? Answer only with the option: '(A) Refuse' or '(B) Comply'.\n"
        if dataset_name == "refusal"
        else "For the following user request and model response, does the answer align with the user's behavior? Answer only with the option: '(A) Agree' or '(B) Disagree'.\n"
    )

    for item in dataset_data:
        question = item["question"]
        matching_answer = item["answer_matching_behavior"]
        not_matching_answer = item["answer_not_matching_behavior"]

        matching_pair = f"{instruction}\n{question}\n{matching_answer}"
        not_matching_pair = f"{instruction}\n{question}\n{not_matching_answer}"

        questions.extend([not_matching_pair, matching_pair])
        labels.extend([0, 1])

    # Get embeddings with smaller batch size to avoid OOM
    print(f"Extracting embeddings for {len(questions)} questions in batches...")
    embeddings = get_sentence_embeddings(
        questions, model_name=args.llm, layer=embedding_layer, batch_size=8  # Reduced batch size
    )
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    print(f"Embeddings shape: {embeddings_tensor.shape}, Labels shape: {labels_tensor.shape}")

    # Load and evaluate SSAE
    from ssae import DictLinearAE

    weights_path = os.path.join(args.modeldir, "weights.pth")
    state_dict = torch.load(weights_path, map_location="cpu")

    rep_dim = state_dict["encoder.weight"].shape[1]
    hid_dim = state_dict["encoder.weight"].shape[0]

    ssae_model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    ssae_model.load_state_dict(state_dict)
    ssae_model.eval()

    # Process SSAE activations in batches to avoid OOM
    batch_size = getattr(args, 'batch_size', 32)
    ssae_acts_list = []
    print(f"Processing SSAE activations in batches of {batch_size}...")

    with torch.no_grad():
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i + batch_size]
            batch_acts = ssae_model.encoder(batch)
            ssae_acts_list.append(batch_acts.cpu())

    ssae_acts = torch.cat(ssae_acts_list, dim=0)
    print(f"SSAE activations shape: {ssae_acts.shape}")

    ssae_corr_vector = compute_correlations(ssae_acts, labels_tensor)

    # Find best feature (max absolute correlation)
    best_feature_idx = ssae_corr_vector.abs().argmax().item()
    ssae_max_mcc = ssae_corr_vector.abs().max().item()

    # Compute MCC for best feature over samples
    best_feature_acts = (ssae_acts[:, best_feature_idx] > 0.1).float()  # Binarize activations
    ssae_mean_mcc, ssae_std_error = compute_mcc_over_samples(best_feature_acts, labels_tensor.squeeze())

    # Load and evaluate appropriate external SAE based on LLM model
    external_name = ""
    external_max_mcc = 0.0
    external_mean_mcc = 0.0
    external_std_error = 0.0

    print(f"\nDetermining appropriate external SAE for LLM: {args.llm}")

    if "pythia" in args.llm.lower():
        # Load Pythia SAE for Pythia models
        print("→ Using Pythia SAE (matches Pythia embedding model)")
        try:
            (
                pythia_decoder_weight,
                pythia_decoder_bias,
                pythia_encoder_weight,
                pythia_encoder_bias,
            ) = load_pythia_sae_checkpoint(embedding_layer)
            pythia_model = ExternalSAEModel(
                pythia_decoder_weight,
                pythia_decoder_bias,
                pythia_encoder_weight,
                pythia_encoder_bias,
            )

            # Process Pythia SAE activations in batches
            pythia_acts_list = []
            print(f"Processing Pythia SAE activations in batches of {batch_size}...")

            with torch.no_grad():
                for i in range(0, len(embeddings_tensor), batch_size):
                    batch = embeddings_tensor[i:i + batch_size]
                    batch_acts = pythia_model.encoder(batch)
                    pythia_acts_list.append(batch_acts.cpu())

            pythia_acts = torch.cat(pythia_acts_list, dim=0)
            print(f"Pythia SAE activations shape: {pythia_acts.shape}")

            pythia_corr_vector = compute_correlations(
                pythia_acts, labels_tensor
            )

            # Find best feature for Pythia SAE
            pythia_best_feature_idx = pythia_corr_vector.abs().argmax().item()
            external_max_mcc = pythia_corr_vector.abs().max().item()

            # Compute MCC for best feature over samples
            pythia_best_feature_acts = (pythia_acts[:, pythia_best_feature_idx] > 0.1).float()
            external_mean_mcc, external_std_error = compute_mcc_over_samples(pythia_best_feature_acts, labels_tensor.squeeze())
            external_name = "Pythia SAE"
        except Exception as e:
            print(f"Failed to load Pythia SAE: {e}")
            external_name = "Failed"
    elif "gemma" in args.llm.lower():
        # Load Gemma Scope for Gemma models
        print("→ Using Gemma Scope (matches Gemma embedding model)")
        try:
            (
                gemma_decoder_weight,
                gemma_decoder_bias,
                gemma_encoder_weight,
                gemma_encoder_bias,
            ) = load_gemmascope_checkpoint()
            gemma_model = ExternalSAEModel(
                gemma_decoder_weight,
                gemma_decoder_bias,
                gemma_encoder_weight,
                gemma_encoder_bias,
            )

            # Process Gemma Scope activations in batches
            gemma_acts_list = []
            print(f"Processing Gemma Scope activations in batches of {batch_size}...")

            with torch.no_grad():
                for i in range(0, len(embeddings_tensor), batch_size):
                    batch = embeddings_tensor[i:i + batch_size]
                    batch_acts = gemma_model.encoder(batch)
                    gemma_acts_list.append(batch_acts.cpu())

            gemma_acts = torch.cat(gemma_acts_list, dim=0)
            print(f"Gemma Scope activations shape: {gemma_acts.shape}")

            gemma_corr_vector = compute_correlations(gemma_acts, labels_tensor)

            # Find best feature for Gemma Scope
            gemma_best_feature_idx = gemma_corr_vector.abs().argmax().item()
            external_max_mcc = gemma_corr_vector.abs().max().item()

            # Compute MCC for best feature over samples
            gemma_best_feature_acts = (gemma_acts[:, gemma_best_feature_idx] > 0.1).float()
            external_mean_mcc, external_std_error = compute_mcc_over_samples(gemma_best_feature_acts, labels_tensor.squeeze())
            external_name = "Gemma Scope"
        except Exception as e:
            print(f"Failed to load Gemma Scope: {e}")
            external_name = "Failed"
    else:
        # Unsupported model
        print(f"→ No matching external SAE found for model: {args.llm}")
        print("→ Supported models: pythia (uses Pythia SAE), gemma (uses Gemma Scope)")
        external_name = "Unsupported"

    # Print simple results
    print(f"\n{'='*50}")
    print(f"MCC COMPARISON RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"LLM Model: {args.llm}")
    print(f"Dataset Samples: {len(labels)}")
    print(f"\nSSAE Statistics:")
    print(f"  Max MCC: {ssae_max_mcc:.4f}")
    print(f"  Mean MCC: {ssae_mean_mcc:.4f} ± {ssae_std_error:.4f}")

    if external_name not in ["Failed", "Unsupported"]:
        print(f"\n{external_name} Statistics:")
        print(f"  Max MCC: {external_max_mcc:.4f}")
        print(f"  Mean MCC: {external_mean_mcc:.4f} ± {external_std_error:.4f}")

        print(f"\nComparison:")
        max_diff = external_max_mcc - ssae_max_mcc
        mean_diff = external_mean_mcc - ssae_mean_mcc
        print(f"  Max MCC difference ({external_name} - SSAE): {max_diff:+.4f}")
        print(f"  Mean MCC difference ({external_name} - SSAE): {mean_diff:+.4f}")

        if abs(mean_diff) < 0.001:
            print(f"→ Similar mean performance")
        elif mean_diff > 0:
            print(f"→ {external_name} has better mean performance")
        else:
            print(f"→ SSAE has better mean performance")
    else:
        print(f"\n{external_name}: Unable to load external SAE")

    return {
        "dataset": dataset_name,
        "ssae": {
            "max_mcc": ssae_max_mcc,
            "mean_mcc": ssae_mean_mcc,
            "std_error": ssae_std_error
        },
        "external_sae": {
            "name": external_name,
            "max_mcc": external_max_mcc if external_name not in ["Failed", "Unsupported"] else None,
            "mean_mcc": external_mean_mcc if external_name not in ["Failed", "Unsupported"] else None,
            "std_error": external_std_error if external_name not in ["Failed", "Unsupported"] else None
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


def compute_mcc_over_samples(predictions, labels, n_bootstrap=100):
    """
    Compute MCC mean and std error over bootstrap samples.

    Args:
        predictions: Binary predictions for the best feature [N]
        labels: Binary labels [N]
        n_bootstrap: Number of bootstrap samples

    Returns:
        mean_mcc, std_error
    """
    from sklearn.metrics import matthews_corrcoef

    n_samples = len(predictions)
    predictions_np = predictions.numpy()
    labels_np = labels.numpy()

    # Bootstrap sampling to get MCC distribution
    mcc_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_preds = predictions_np[indices]
        boot_labels = labels_np[indices]

        # Compute MCC for this bootstrap sample
        mcc = matthews_corrcoef(boot_labels, boot_preds)
        mcc_scores.append(mcc)

    mcc_scores = np.array(mcc_scores)
    mean_mcc = np.mean(mcc_scores)
    std_error = np.std(mcc_scores) / np.sqrt(n_bootstrap)

    return mean_mcc, std_error


def get_steering_vector(args, dataset_name="refusal"):
    """Load dataset and find the best steering vector dimension using correlation analysis.
    This function maintains backward compatibility for generation functionality.
    """

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
    steering_vector = decoder_weight[
        :, ssae_results["best_feature_idx"]
    ]  # Shape: (rep_dim,)

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

            with open(args.output_json, "w") as f:
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing activations (default: 32, reduce if OOM)",
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
