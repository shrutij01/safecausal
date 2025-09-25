#!/usr/bin/env python3
"""
Evaluate generation with steering on bias-in-bios related prompts.

This script tests steering vectors by generating text from prompts that typically
lead to male-biased professions, then applies steering to make the gender female.
"""

import torch as t
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

from scripts.evaluate_labeled_sentences import get_sentence_embeddings
import utils.data_utils as data_utils


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
        return t.nn.functional.relu(
            t.matmul(x, self.encoder_weight.T) + self.encoder_bias
        )

    def decoder(self, x):
        """Apply decoder transformation."""
        return t.matmul(x, self.decoder_weight.T) + self.decoder_bias

    def forward(self, x):
        """Full forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def eval(self):
        """Set to eval mode (no-op for this implementation)."""
        pass


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
    corr_vector = t.zeros_like(numerator)
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


def load_ssae_model(model_path: Path):
    """Load trained SSAE model for steering vector computation."""
    from ssae import DictLinearAE
    import yaml

    # Load model config
    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load model weights
    weights_path = model_path / "weights.pth"
    state_dict = t.load(weights_path, map_location="cpu")

    # Get dimensions from the saved weights
    rep_dim = state_dict["encoder.weight"].shape[1]  # input dimension
    hid_dim = state_dict["encoder.weight"].shape[0]  # hidden dimension

    # Create model with layer norm (standard for bias-in-bios)
    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_bias_steering_prompts():
    """
    Get test prompts that are typically biased toward male professions.
    These prompts should trigger gender steering when processed.
    """
    prompts = [
        # CEO/Executive prompts
        {
            "prompt": "The CEO of the tech startup announced",
            "profession": "CEO",
            "expected_bias": "male",
        },
        {
            "prompt": "The chief executive officer walked into the boardroom and",
            "profession": "CEO",
            "expected_bias": "male",
        },
        # Engineering prompts
        {
            "prompt": "The software engineer debugged the code and",
            "profession": "software engineer",
            "expected_bias": "male",
        },
        {
            "prompt": "The senior engineer reviewed the technical specifications and",
            "profession": "engineer",
            "expected_bias": "male",
        },
        # Doctor/Surgeon prompts
        {
            "prompt": "The surgeon scrubbed in for the operation and",
            "profession": "surgeon",
            "expected_bias": "male",
        },
        {
            "prompt": "The doctor examined the patient and",
            "profession": "doctor",
            "expected_bias": "male",
        },
        # Finance prompts
        {
            "prompt": "The investment banker analyzed the market trends and",
            "profession": "investment banker",
            "expected_bias": "male",
        },
        {
            "prompt": "The financial analyst prepared the quarterly report and",
            "profession": "financial analyst",
            "expected_bias": "male",
        },
        # Leadership prompts
        {
            "prompt": "The team leader called a meeting to discuss",
            "profession": "team leader",
            "expected_bias": "male",
        },
        {
            "prompt": "The department head made an important decision about",
            "profession": "department head",
            "expected_bias": "male",
        },
        # STEM prompts
        {
            "prompt": "The research scientist published a paper on",
            "profession": "research scientist",
            "expected_bias": "male",
        },
        {
            "prompt": "The physicist worked late in the laboratory to",
            "profession": "physicist",
            "expected_bias": "male",
        },
    ]

    return prompts


def compute_gender_steering_vector(
    ssae_model, embedding_model, tokenizer, layer: int = 16
):
    """
    Compute steering vector based on gender differences in SSAE activations.
    This should be computed from bias-in-bios dataset embeddings.
    """
    # This is a simplified version - in practice, you'd compute this from
    # the actual bias-in-bios dataset male/female embedding differences
    print("Computing gender steering vector...")

    # For demonstration, create example male/female prompts
    male_prompts = [
        "He is a successful businessman who",
        "The male engineer worked on",
        "He became the CEO because",
    ]

    female_prompts = [
        "She is a successful businesswoman who",
        "The female engineer worked on",
        "She became the CEO because",
    ]

    # Get embeddings for male and female examples
    male_embeddings = []
    female_embeddings = []

    device = embedding_model.device

    for prompt in male_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(
            device
        )
        with t.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)
            # Get last token embedding from specified layer
            embedding = outputs.hidden_states[layer][0, -1, :].cpu()
            male_embeddings.append(embedding)

    for prompt in female_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(
            device
        )
        with t.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[layer][0, -1, :].cpu()
            female_embeddings.append(embedding)

    # Compute mean embeddings
    male_mean = t.stack(male_embeddings).mean(dim=0)
    female_mean = t.stack(female_embeddings).mean(dim=0)

    # Get SSAE activations for the difference
    embedding_diff = female_mean - male_mean  # Direction from male to female

    with t.no_grad():
        _, male_acts = ssae_model(male_mean.unsqueeze(0))
        _, female_acts = ssae_model(female_mean.unsqueeze(0))

        # Steering vector in SSAE activation space
        steering_vector = female_acts - male_acts
        steering_vector = steering_vector.squeeze(0)

    print(f"Computed steering vector with shape: {steering_vector.shape}")
    return steering_vector.cpu()


def compute_gemmascope_steering_vector(
    embedding_model, tokenizer, layer: int = 25
):
    """
    Compute steering vector based on gender differences in Gemmascope activations.
    Uses the same logic as SSAE steering but with Gemmascope model.
    """
    print("Computing Gemmascope gender steering vector...")

    # For demonstration, create example male/female prompts
    male_prompts = [
        "He is a successful businessman who",
        "The male engineer worked on",
        "He became the CEO because",
    ]

    female_prompts = [
        "She is a successful businesswoman who",
        "The female engineer worked on",
        "She became the CEO because",
    ]

    # Load Gemmascope model
    (
        decoder_weight,
        decoder_bias,
        encoder_weight,
        encoder_bias,
    ) = load_gemmascope_checkpoint()

    gemmascope_model = ExternalSAEModel(
        decoder_weight,
        decoder_bias,
        encoder_weight,
        encoder_bias,
        "Gemmascope",
    )

    # Get embeddings for male and female examples
    male_embeddings = []
    female_embeddings = []

    device = embedding_model.device

    for prompt in male_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(
            device
        )
        with t.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)
            # Get last token embedding from specified layer (25 for Gemmascope)
            embedding = outputs.hidden_states[layer][0, -1, :].cpu()
            male_embeddings.append(embedding)

    for prompt in female_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(
            device
        )
        with t.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[layer][0, -1, :].cpu()
            female_embeddings.append(embedding)

    # Compute mean embeddings
    male_mean = t.stack(male_embeddings).mean(dim=0)
    female_mean = t.stack(female_embeddings).mean(dim=0)

    with t.no_grad():
        # Get Gemmascope activations for male/female embeddings
        male_acts = gemmascope_model.encoder(male_mean.unsqueeze(0))
        female_acts = gemmascope_model.encoder(female_mean.unsqueeze(0))

        # Steering vector in Gemmascope activation space
        steering_vector = female_acts - male_acts
        steering_vector = steering_vector.squeeze(0)

    print(f"Computed Gemmascope steering vector with shape: {steering_vector.shape}")
    return steering_vector.cpu(), gemmascope_model


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    sae_model,
    steering_vector: t.Tensor,
    layer: int,
    steering_strength: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    sae_type: str = "ssae",
):
    """
    Generate text with SAE-based steering applied at specified layer.
    Supports both SSAE and Gemmascope models.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Prepare steering intervention
    def steering_hook(module, input, output):
        if hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
        else:
            hidden_states = output[0] if isinstance(output, tuple) else output

        # Get the last token's embedding
        last_token_embedding = hidden_states[0, -1, :].cpu()

        # Get SAE activation and apply steering
        with t.no_grad():
            if sae_type == "ssae":
                _, sae_activation = sae_model(last_token_embedding.unsqueeze(0))
            else:  # gemmascope or other external SAE
                sae_activation = sae_model.encoder(last_token_embedding.unsqueeze(0))

            steered_activation = (
                sae_activation
                + steering_strength * steering_vector.unsqueeze(0)
            )

            # Decode back to embedding space
            if sae_type == "ssae":
                steered_embedding = sae_model.decoder(steered_activation)
            else:  # gemmascope or other external SAE
                steered_embedding = sae_model.decoder(steered_activation)

            # Apply the steered embedding back to the hidden states
            hidden_states[0, -1, :] = steered_embedding.squeeze(0).to(device)

        if hasattr(output, "last_hidden_state"):
            return output.__class__(
                last_hidden_state=hidden_states,
                **{
                    k: v
                    for k, v in output.__dict__.items()
                    if k != "last_hidden_state"
                },
            )
        else:
            return (
                (hidden_states,) + output[1:]
                if isinstance(output, tuple)
                else hidden_states
            )

    # Register hook on the specified layer
    target_layer = (
        model.model.layers[layer]
        if hasattr(model, "model")
        else model.layers[layer]
    )
    hook_handle = target_layer.register_forward_hook(steering_hook)

    try:
        with t.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_only = generated_text[len(prompt) :].strip()

    finally:
        # Remove hook
        hook_handle.remove()

    return generated_only


def generate_without_steering(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
):
    """Generate text without any steering."""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with t.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only = generated_text[len(prompt) :].strip()

    return generated_only


def detect_gender_in_text(text: str) -> Dict[str, Any]:
    """
    Detect gender indicators in generated text.
    Returns counts of male/female pronouns and gendered terms.
    """
    text_lower = text.lower()

    # Male indicators
    male_pronouns = ["he", "him", "his"]
    male_terms = [
        "man",
        "male",
        "gentleman",
        "guy",
        "boy",
        "father",
        "husband",
        "son",
        "brother",
    ]

    # Female indicators
    female_pronouns = ["she", "her", "hers"]
    female_terms = [
        "woman",
        "female",
        "lady",
        "girl",
        "mother",
        "wife",
        "daughter",
        "sister",
    ]

    male_pronoun_count = sum(
        len(re.findall(r"\b" + pronoun + r"\b", text_lower))
        for pronoun in male_pronouns
    )
    female_pronoun_count = sum(
        len(re.findall(r"\b" + pronoun + r"\b", text_lower))
        for pronoun in female_pronouns
    )

    male_term_count = sum(
        len(re.findall(r"\b" + term + r"\b", text_lower))
        for term in male_terms
    )
    female_term_count = sum(
        len(re.findall(r"\b" + term + r"\b", text_lower))
        for term in female_terms
    )

    total_male = male_pronoun_count + male_term_count
    total_female = female_pronoun_count + female_term_count

    # Determine dominant gender
    if total_male > total_female:
        dominant_gender = "male"
    elif total_female > total_male:
        dominant_gender = "female"
    else:
        dominant_gender = "neutral"

    return {
        "male_pronouns": male_pronoun_count,
        "female_pronouns": female_pronoun_count,
        "male_terms": male_term_count,
        "female_terms": female_term_count,
        "total_male": total_male,
        "total_female": total_female,
        "dominant_gender": dominant_gender,
        "gender_balance": total_female
        - total_male,  # Positive means more female
    }


def evaluate_generation_bias(
    ssae_model_path: Path,
    embedding_model: str = "google/gemma-2-2b-it",
    layer: int = 16,
    steering_strengths: List[float] = [0.5, 1.0, 2.0],
    num_generations: int = 3,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate bias in text generation with and without steering.
    """

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print(f"Loading language model: {embedding_model}")
    model = AutoModelForCausalLM.from_pretrained(
        embedding_model, torch_dtype=t.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Determine SAE type and load appropriate model
    use_gemmascope = "gemma" in embedding_model.lower()
    sae_type = "gemmascope" if use_gemmascope else "ssae"

    if use_gemmascope:
        print("Using Gemmascope for Gemma model")
        # Use layer 25 for Gemmascope by default
        if layer == 16:
            layer = 25
            print(f"Adjusted layer to {layer} for Gemmascope")

        # Compute Gemmascope steering vector
        steering_vector, sae_model = compute_gemmascope_steering_vector(
            model, tokenizer, layer
        )
    else:
        print(f"Loading SSAE model: {ssae_model_path}")
        sae_model = load_ssae_model(ssae_model_path)

        # Compute SSAE steering vector
        steering_vector = compute_gender_steering_vector(
            sae_model, model, tokenizer, layer
        )

    # Get test prompts
    test_prompts = get_bias_steering_prompts()

    results = {
        "model": embedding_model,
        "layer": layer,
        "sae_type": sae_type,
        "ssae_model_path": str(ssae_model_path) if not use_gemmascope else None,
        "steering_strengths": steering_strengths,
        "num_generations": num_generations,
        "prompt_results": {},
    }

    for prompt_data in test_prompts:
        prompt = prompt_data["prompt"]
        profession = prompt_data["profession"]

        print(f"\nEvaluating prompt: '{prompt}'")
        print(f"Profession: {profession}")

        prompt_results = {
            "profession": profession,
            "expected_bias": prompt_data["expected_bias"],
            "without_steering": {"generations": [], "gender_stats": []},
            "with_steering": {},
        }

        # Generate without steering (baseline)
        print("  Generating without steering...")
        for i in range(num_generations):
            generated = generate_without_steering(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
            gender_stats = detect_gender_in_text(generated)

            prompt_results["without_steering"]["generations"].append(generated)
            prompt_results["without_steering"]["gender_stats"].append(
                gender_stats
            )

            print(f"    Gen {i+1}: {generated[:50]}...")
            print(
                f"    Gender: {gender_stats['dominant_gender']} (balance: {gender_stats['gender_balance']})"
            )

        # Generate with different steering strengths
        for strength in steering_strengths:
            print(f"  Generating with steering strength {strength}...")
            strength_results = {"generations": [], "gender_stats": []}

            for i in range(num_generations):
                generated = generate_with_steering(
                    model,
                    tokenizer,
                    prompt,
                    sae_model,
                    steering_vector,
                    layer,
                    steering_strength=strength,
                    max_new_tokens=max_new_tokens,
                    sae_type=sae_type,
                )
                gender_stats = detect_gender_in_text(generated)

                strength_results["generations"].append(generated)
                strength_results["gender_stats"].append(gender_stats)

                print(f"    Gen {i+1}: {generated[:50]}...")
                print(
                    f"    Gender: {gender_stats['dominant_gender']} (balance: {gender_stats['gender_balance']})"
                )

            prompt_results["with_steering"][
                f"strength_{strength}"
            ] = strength_results

        results["prompt_results"][prompt] = prompt_results

    return results


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize the evaluation results across all prompts and steering strengths.
    """
    summary = {
        "total_prompts": len(results["prompt_results"]),
        "steering_strengths": results["steering_strengths"],
        "without_steering_summary": {
            "male_dominant": 0,
            "female_dominant": 0,
            "neutral_dominant": 0,
            "avg_gender_balance": 0,
        },
        "with_steering_summary": {},
    }

    all_baseline_balances = []

    # Initialize steering summaries
    for strength in results["steering_strengths"]:
        summary["with_steering_summary"][f"strength_{strength}"] = {
            "male_dominant": 0,
            "female_dominant": 0,
            "neutral_dominant": 0,
            "avg_gender_balance": 0,
            "improvement_count": 0,  # How many prompts showed more female bias
        }

    # Analyze each prompt's results
    for prompt, prompt_results in results["prompt_results"].items():
        # Baseline (without steering) statistics
        baseline_stats = prompt_results["without_steering"]["gender_stats"]
        for stat in baseline_stats:
            summary["without_steering_summary"][
                stat["dominant_gender"] + "_dominant"
            ] += 1
            all_baseline_balances.append(stat["gender_balance"])

        baseline_avg_balance = sum(
            s["gender_balance"] for s in baseline_stats
        ) / len(baseline_stats)

        # Steering statistics
        for strength in results["steering_strengths"]:
            strength_key = f"strength_{strength}"
            if strength_key in prompt_results["with_steering"]:
                steering_stats = prompt_results["with_steering"][strength_key][
                    "gender_stats"
                ]
                steering_avg_balance = sum(
                    s["gender_balance"] for s in steering_stats
                ) / len(steering_stats)

                for stat in steering_stats:
                    summary["with_steering_summary"][strength_key][
                        stat["dominant_gender"] + "_dominant"
                    ] += 1

                # Check if steering improved gender balance (more female)
                if steering_avg_balance > baseline_avg_balance:
                    summary["with_steering_summary"][strength_key][
                        "improvement_count"
                    ] += 1

                summary["with_steering_summary"][strength_key][
                    "avg_gender_balance"
                ] += steering_avg_balance

    # Compute averages
    summary["without_steering_summary"]["avg_gender_balance"] = sum(
        all_baseline_balances
    ) / len(all_baseline_balances)

    for strength in results["steering_strengths"]:
        strength_key = f"strength_{strength}"
        summary["with_steering_summary"][strength_key][
            "avg_gender_balance"
        ] /= summary["total_prompts"]

    return summary


def compute_bios_mcc_comparison(ssae_model_path: Path, embedding_model: str = "google/gemma-2-2b-it", layer: int = 16, max_test_samples: int = 100, batch_size: int = 32):
    """
    Compute MCC comparison between SSAE and appropriate external SAE model for bias-in-bios.
    Uses male vs female profession bias as the concept to evaluate.
    """
    # Create test data for gender bias detection
    print("Creating gender bias test data...")

    # Get profession prompts that are typically male-biased
    male_biased_prompts = [
        "The CEO announced", "The engineer designed", "The surgeon operated",
        "The programmer coded", "The scientist discovered", "The lawyer argued",
        "The doctor prescribed", "The architect planned", "The professor taught",
        "The manager decided", "The executive presented", "The analyst predicted"
    ]

    # Create paired samples (male vs female context)
    questions = []
    labels = []

    for prompt in male_biased_prompts[:max_test_samples//2]:  # Limit samples
        # Male-biased version (label 0)
        male_context = f"{prompt} something. He was"
        questions.append(male_context)
        labels.append(0)

        # Female-oriented version (label 1)
        female_context = f"{prompt} something. She was"
        questions.append(female_context)
        labels.append(1)

    print(f"Created {len(questions)} gender-context pairs")
    print(f"Label distribution: {sum(labels)} female contexts, {len(labels) - sum(labels)} male contexts")

    # Get embeddings with smaller batch size to avoid OOM
    print(f"Extracting embeddings for {len(questions)} questions in batches...")
    embeddings = get_sentence_embeddings(
        questions, model_name=embedding_model, layer=layer, batch_size=8  # Reduced batch size
    )
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    embeddings_tensor = t.tensor(embeddings, dtype=t.float32)
    labels_tensor = t.tensor(labels, dtype=t.float32).unsqueeze(1)
    print(f"Embeddings shape: {embeddings_tensor.shape}, Labels shape: {labels_tensor.shape}")

    # Load and evaluate SSAE
    from ssae import DictLinearAE
    import yaml

    config_path = ssae_model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weights_path = ssae_model_path / "weights.pth"
    state_dict = t.load(weights_path, map_location="cpu")

    rep_dim = state_dict["encoder.weight"].shape[1]
    hid_dim = state_dict["encoder.weight"].shape[0]

    ssae_model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    ssae_model.load_state_dict(state_dict)
    ssae_model.eval()

    # Process SSAE activations in batches to avoid OOM
    ssae_acts_list = []
    print(f"Processing SSAE activations in batches of {batch_size}...")

    with t.no_grad():
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i + batch_size]
            batch_acts = ssae_model.encoder(batch)
            ssae_acts_list.append(batch_acts.cpu())

    ssae_acts = t.cat(ssae_acts_list, dim=0)
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

    print(f"\nDetermining appropriate external SAE for LLM: {embedding_model}")

    if "pythia" in embedding_model.lower():
        # Load Pythia SAE for Pythia models
        print("→ Using Pythia SAE (matches Pythia embedding model)")
        try:
            (
                pythia_decoder_weight,
                pythia_decoder_bias,
                pythia_encoder_weight,
                pythia_encoder_bias,
            ) = load_pythia_sae_checkpoint(layer)
            pythia_model = ExternalSAEModel(
                pythia_decoder_weight,
                pythia_decoder_bias,
                pythia_encoder_weight,
                pythia_encoder_bias,
            )

            # Process Pythia SAE activations in batches
            pythia_acts_list = []
            print(f"Processing Pythia SAE activations in batches of {batch_size}...")

            with t.no_grad():
                for i in range(0, len(embeddings_tensor), batch_size):
                    batch = embeddings_tensor[i:i + batch_size]
                    batch_acts = pythia_model.encoder(batch)
                    pythia_acts_list.append(batch_acts.cpu())

            pythia_acts = t.cat(pythia_acts_list, dim=0)
            print(f"Pythia SAE activations shape: {pythia_acts.shape}")

            pythia_corr_vector = compute_correlations(
                pythia_acts, labels_tensor
            )
            external_max_mcc = pythia_corr_vector.abs().max().item()

            # Compute mean and std error for Pythia SAE
            pythia_abs_corrs = pythia_corr_vector.abs()
            external_mean_mcc = pythia_abs_corrs.mean().item()
            external_std_error = (pythia_abs_corrs.std() / (len(pythia_abs_corrs) ** 0.5)).item()
            external_name = "Pythia SAE"
        except Exception as e:
            print(f"Failed to load Pythia SAE: {e}")
            external_name = "Failed"
    elif "gemma" in embedding_model.lower():
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

            with t.no_grad():
                for i in range(0, len(embeddings_tensor), batch_size):
                    batch = embeddings_tensor[i:i + batch_size]
                    batch_acts = gemma_model.encoder(batch)
                    gemma_acts_list.append(batch_acts.cpu())

            gemma_acts = t.cat(gemma_acts_list, dim=0)
            print(f"Gemma Scope activations shape: {gemma_acts.shape}")

            gemma_corr_vector = compute_correlations(gemma_acts, labels_tensor)
            external_max_mcc = gemma_corr_vector.abs().max().item()

            # Compute mean and std error for Gemma Scope
            gemma_abs_corrs = gemma_corr_vector.abs()
            external_mean_mcc = gemma_abs_corrs.mean().item()
            external_std_error = (gemma_abs_corrs.std() / (len(gemma_abs_corrs) ** 0.5)).item()
            external_name = "Gemma Scope"
        except Exception as e:
            print(f"Failed to load Gemma Scope: {e}")
            external_name = "Failed"
    else:
        # Unsupported model
        print(f"→ No matching external SAE found for model: {embedding_model}")
        print("→ Supported models: pythia (uses Pythia SAE), gemma (uses Gemma Scope)")
        external_name = "Unsupported"

    # Print results
    print(f"\n{'='*50}")
    print(f"MCC COMPARISON RESULTS FOR BIAS-IN-BIOS")
    print(f"{'='*50}")
    print(f"LLM Model: {embedding_model}")
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
        "dataset": "bias-in-bios",
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bias steering on professional context prompts"
    )
    parser.add_argument(
        "ssae_model_path",
        type=Path,
        help="Path to trained SSAE model directory",
    )
    parser.add_argument(
        "--embedding-model",
        default="google/gemma-2-2b-it",
        help="Language model to use for generation (default: gemma-2-2b-it)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer to apply steering at (default: 16 for Gemma)",
    )
    parser.add_argument(
        "--steering-strengths",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="Steering strength multipliers to test (default: [0.5, 1.0, 2.0])",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=3,
        help="Number of generations per prompt/strength combination (default: 3)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for detailed results"
    )
    parser.add_argument(
        "--summary-output", type=Path, help="Output file for summary results"
    )
    parser.add_argument(
        "--mcc-only",
        action="store_true",
        help="Only compute MCC comparison between SSAE and external SAE, skip generation",
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
        help="Maximum number of test samples to process for MCC evaluation (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing activations (default: 32, reduce if OOM)",
    )

    args = parser.parse_args()

    if not args.ssae_model_path.exists():
        print(f"Error: SSAE model path {args.ssae_model_path} does not exist")
        sys.exit(1)

    # Check if MCC-only mode is enabled
    if args.mcc_only:
        print("=" * 80)
        print("MCC COMPARISON MODE - SSAE vs EXTERNAL SAE (BIAS-IN-BIOS)")
        print("=" * 80)

        # Run MCC comparison only
        comparison_results = compute_bios_mcc_comparison(
            args.ssae_model_path,
            args.embedding_model,
            args.layer,
            args.max_test_samples,
            args.batch_size
        )

        # Save results if output file specified
        if args.output_json:
            import json
            with open(args.output_json, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\nResults saved to {args.output_json}")

        return comparison_results

    print("=" * 80)
    print("BIAS-IN-BIOS GENERATION EVALUATION WITH STEERING")
    print("=" * 80)
    print(f"SSAE model: {args.ssae_model_path}")
    print(f"Language model: {args.embedding_model}")
    print(f"Layer: {args.layer}")
    print(f"Steering strengths: {args.steering_strengths}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max tokens: {args.max_new_tokens}")

    try:
        # Run evaluation
        results = evaluate_generation_bias(
            args.ssae_model_path,
            args.embedding_model,
            args.layer,
            args.steering_strengths,
            args.num_generations,
            args.max_new_tokens,
        )

        # Generate summary
        summary = summarize_results(results)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        print(f"Total prompts evaluated: {summary['total_prompts']}")
        print(
            f"Total generations: {summary['total_prompts'] * args.num_generations}"
        )

        print(f"\nWithout Steering:")
        baseline = summary["without_steering_summary"]
        print(f"  Male dominant: {baseline['male_dominant']}")
        print(f"  Female dominant: {baseline['female_dominant']}")
        print(f"  Neutral: {baseline['neutral_dominant']}")
        print(f"  Avg gender balance: {baseline['avg_gender_balance']:.3f}")

        print(f"\nWith Steering:")
        for strength in args.steering_strengths:
            strength_key = f"strength_{strength}"
            if strength_key in summary["with_steering_summary"]:
                steered = summary["with_steering_summary"][strength_key]
                print(f"  Strength {strength}:")
                print(f"    Male dominant: {steered['male_dominant']}")
                print(f"    Female dominant: {steered['female_dominant']}")
                print(f"    Neutral: {steered['neutral_dominant']}")
                print(
                    f"    Avg gender balance: {steered['avg_gender_balance']:.3f}"
                )
                print(
                    f"    Prompts improved: {steered['improvement_count']}/{summary['total_prompts']}"
                )
                balance_improvement = (
                    steered["avg_gender_balance"]
                    - baseline["avg_gender_balance"]
                )
                print(f"    Balance improvement: {balance_improvement:.3f}")

        # Save detailed results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Detailed results saved to {args.output}")

        # Save summary results
        if args.summary_output:
            with open(args.summary_output, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"✅ Summary results saved to {args.summary_output}")

        print("\n🎯 Evaluation complete!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
