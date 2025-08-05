from json import decoder
from re import L
import data_utils as utils
import metrics

from ssae import DictLinearAE

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from box import Box
from collections import Counter, defaultdict
import debug_tools as dbg

import itertools
import os
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from nnsight import CONFIG, LanguageModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG.API.APIKEY = "224de021-0457-4848-b96d-487af632c352"


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


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


def pca_transform(X: torch.Tensor):
    """
    Computes the first principal direction of X.

    Args:
        X (torch.Tensor): Input data of shape (n_samples, n_features)

    Returns:
        direction (torch.Tensor): First principal direction (unit vector of shape [n_features])
    """
    n, d = X.shape
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean
    U, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    U, Vh = svd_flip(U, Vh)
    components = Vh
    return torch.matmul(X - mean, components.t()), components, mean


def load_llamascope_checkpoint():
    model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"

    filename = "Llama3_1-8B-Base-L31R-8x/checkpoints/final.safetensors"

    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded checkpoint to: {filepath}")

    # 2. Load the safetensor
    state_dict = load_file(filepath)
    decoder_weight = state_dict[
        "decoder.weight"
    ]  # shape: (vocab_size, d_model)
    decoder_bias = state_dict["decoder.bias"]
    encoder_weight = state_dict["encoder.weight"]
    encoder_bias = state_dict["encoder.bias"]
    print(f"decoder.weight shape: {decoder_weight.shape}")

    return (decoder_weight, decoder_bias, encoder_weight, encoder_bias)


def get_concept_detection_logits(
    z, encoder_weight, encoder_bias, decoder_bias, args
):
    """
    Compute the logits for 1sp concept detection.

    Args:
        concept_projections: [B, C] tensor of concept projections

    Returns:
        logits: [B, C] tensor of logits
    """
    encoder_weight = encoder_weight.to(z.device)
    encoder_bias = encoder_bias.to(z.device)
    decoder_bias = decoder_bias.to(z.device)
    if args.modeltype == "llamascope":
        concept_projections = torch.nn.functional.relu(
            encoder_weight.to(torch.float32) @ z.T + encoder_bias.unsqueeze(1)
        ).T
    elif args.modeltype == "ssae":
        concept_projections = (
            encoder_weight.to(torch.float32)
            @ (z.T - decoder_bias.unsqueeze(1))  # enc_dim x batch_size
            + encoder_bias.unsqueeze(1)
        ).T  # batch_size x enc_dim
    else:
        raise ValueError
    concept_detection_activations = concept_projections.max(dim=1).values
    concept_detection_scores = (
        concept_detection_activations
        - concept_detection_activations.min(dim=0).values
    ) / (
        concept_detection_activations.max(dim=0).values
        - concept_detection_activations.min(dim=0).values
    )
    # concept_projections = (concept_projections > 0.1).float()
    eps = 1e-10
    concept_logits = torch.nn.functional.softmax(concept_projections, dim=1)

    # Element-wise multiplication of probabilities with their log
    entropy = -torch.sum(
        concept_logits * torch.log(concept_logits + eps), dim=1
    )
    mean_over_max_logits = concept_logits.max(dim=1).values.mean(dim=0)
    print("mean_over_max_logits", mean_over_max_logits)
    print("entropy", entropy.max(), entropy.min(), entropy.mean())
    return (mean_over_max_logits, entropy)


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


def check_model_config_consistency(model_configs: list[Box]) -> None:
    """
    Check that all model configs share the same critical values.
    """
    ref = model_configs[0]
    for i, cfg in enumerate(model_configs[1:], start=1):
        assert (
            cfg.overcompleteness_factor == ref.overcompleteness_factor
        ), f"Model {i+1} has a different overcompleteness_factor: {cfg.overcompleteness_factor} != {ref.overcompleteness_factor}"
        assert (
            cfg.primal_lr == ref.primal_lr
        ), f"Model {i+1} has a different primal_lr: {cfg.primal_lr} != {ref.primal_lr}"
        assert (
            cfg.norm_type == ref.norm_type
        ), f"Model {i+1} has a different norm_type: {cfg.norm_type} != {ref.norm_type}"


def compute_all_pairwise_mccs(
    weight_matrices: list[torch.Tensor],
) -> list[float]:
    """
    Compute mean correlation coefficients (MCCs) between all model pairs.
    """
    mccs = []
    for i, j in itertools.combinations(range(len(weight_matrices)), 2):
        mcc = metrics.mean_corr_coef(
            utils.numpify(weight_matrices[i]),
            utils.numpify(weight_matrices[j]),
            method="pearson",
        )
        mccs.append(mcc)
    return mccs


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


def get_ood_cosine_similarity(steering_vector, ood_data, decoder_bias):
    """
    Compute cosine similarity between steering vector and OOD data.
    """
    tilde_z_ood, z_ood = utils.load_test_data(
        datafile=ood_data,
    )
    tilde_z_ood = utils.tensorify(tilde_z_ood, device)
    tilde_z_ood = F.normalize(tilde_z_ood, dim=1)
    z_ood = utils.tensorify(z_ood, device)
    z_ood = F.normalize(z_ood, dim=1)
    decoder_bias = decoder_bias.to(z_ood.device)
    # steering vector is alreayd normalized
    tilde_hat_z_ood = (
        z_ood + steering_vector.unsqueeze(0) + decoder_bias.unsqueeze(0)
    )
    cosines = []
    tilde_hat_z_ood = utils.numpify(tilde_hat_z_ood)
    tilde_z_ood = utils.numpify(tilde_z_ood)
    for i in range(tilde_z_ood.shape[0]):
        cosines.append(
            cosine_similarity(
                tilde_z_ood[i].reshape(1, -1),
                tilde_hat_z_ood[i].reshape(1, -1),
            )
        )
    print("OOD Cosine Similarities:", np.mean(cosines), np.std(cosines))


def take_pca(z_test, tilde_z_test):
    shifts = utils.tensorify((tilde_z_test - z_test), device)

    shifts_transformed, components, mean = pca_transform(shifts.float())
    pca_vec = (
        (components.sum(dim=0, keepdim=True) + mean).mean(0)
        # .view(z_test[0].shape[0], z_test[0].shape[1])
    )
    z_test = utils.tensorify(z_test, device)
    z_pca = F.normalize(z_test) + pca_vec
    z_pca = F.normalize(z_pca)
    z_pca = utils.numpify(z_pca)
    cosines_pca = []
    for i in range(tilde_z_test.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z_test[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )
    return cosines_pca


def split_test_data_by_concept(tilde_z_test, z_test, concept_labels_test):
    concept_labels_test = np.array(concept_labels_test)

    # Get unique labels
    unique_labels = np.unique(concept_labels_test)

    # Initialize the dictionary
    split_tensors = {}
    # Efficiently gather indices for each label
    for label in unique_labels:
        mask = concept_labels_test == label
        indices = np.where(mask)[0]
        split_tensors[label] = (tilde_z_test[indices], z_test[indices])

    return split_tensors


def compare_top_tokens_with_steering_batch(
    model_name: str,
    input_texts: list[str],
    steering_vector: torch.Tensor,
    layer_idx: int = 16,
    alpha: float = 5.0,
    debug: bool = True,
) -> dict:
    """
    Vectorized comparison of top tokens from original vs steered embeddings using nnsight.

    Args:
        model_name: Hugging Face model name (e.g., "meta-llama/Meta-Llama-3.1-8B")
        input_texts: List of input prompts to process in batch
        steering_vector: Vector to add for steering (shape: [hidden_dim])
        layer_idx: Layer to apply steering intervention
        alpha: Steering strength multiplier
        debug: Enable debug prints

    Returns:
        Dict with batch results: {'original': [tokens_per_input], 'steered': [tokens_per_input]}
    """
    if debug:
        print(f"\n🔧 DEBUG: Starting vectorized steering comparison")
        print(f"📝 Input batch size: {len(input_texts)}")
        print(f"🎯 Layer index: {layer_idx}")
        print(f"💪 Alpha (steering strength): {alpha}")
        print(f"📊 Steering vector shape: {steering_vector.shape}")
        print(f"📊 Steering vector norm: {steering_vector.norm():.4f}")

    # Load model with nnsight
    llm = LanguageModel(model_name, device_map="auto")
    model_device = next(llm.model.parameters()).device

    if debug:
        print(f"🖥️  Model device: {model_device}")
        print(f"🏗️  Model layers: {len(llm.model.layers)}")
        print(f"📍 Original steering vector device: {steering_vector.device}")
        print(f"📍 Steering vector has values: {not steering_vector.is_meta}")

        # Keep steering vector on original device - don't move to meta device
        print(
            f"🔧 Keeping steering vector on original device: {steering_vector.device}"
        )

    # Store results for both conditions
    results = {"original": [], "steered": []}

    # Process batch - get original outputs first
    if debug:
        print(
            f"\n🚀 Running ORIGINAL forward pass for batch of {len(input_texts)}..."
        )

    with llm.trace(input_texts):
        # Save original outputs
        original_outputs = llm.output.save()

    # Extract original logits and decode top tokens
    original_logits = original_outputs[
        "logits"
    ]  # [batch_size, seq_len, vocab_size]
    original_last_logits = original_logits[
        :, -1, :
    ]  # [batch_size, vocab_size]

    # Get top token for each input in batch
    original_top_tokens = []
    for i, logits in enumerate(original_last_logits):
        probs = F.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, 1)
        token = llm.tokenizer.decode(
            [top_idx.item()], skip_special_tokens=True
        )
        original_top_tokens.append((token, top_prob.item()))

    results["original"] = original_top_tokens

    if debug:
        print(
            f"📈 Original top tokens: {original_top_tokens[:3]}..."
        )  # Show first 3

    # Process batch - get steered outputs
    if debug:
        print(
            f"\n🚀 Running STEERED forward pass for batch of {len(input_texts)}..."
        )

    with llm.trace(input_texts):
        # Apply steering to specified layer's output for last token of each sequence
        layer_output = llm.model.layers[layer_idx].output

        if debug:
            print(f"🎯 Applying steering to layer {layer_idx}")

        # Get the hidden states - shape will be [batch_size, seq_len, hidden_dim]
        hidden_states = layer_output[0]

        if debug:
            print(f"📍 Hidden states device: {hidden_states.device}")
            print(f"📍 Hidden states shape: {hidden_states.shape}")
            print(f"📍 Last token shape: {hidden_states[:, -1, :].shape}")

        # Move steering vector to the same device as hidden states (actual GPU device)
        # actual_device = hidden_states.device
        # steering_vec = steering_vector.to(actual_device)

        # Apply steering to last token position for all sequences in batch
        # hidden_states[:, -1, :] has shape [batch_size, hidden_dim]
        # steering_vec has shape [hidden_dim]
        # Broadcasting will add steering_vec to each sequence's last token
        original_last_hidden = hidden_states[:, -1, :].clone()
        hidden_states[:, -1, :] = (
            hidden_states[:, -1, :] + alpha * steering_vector
        )

        if debug:
            steered_last_hidden = hidden_states[:, -1, :]
            change_magnitude = (
                (steered_last_hidden - original_last_hidden)
                .norm(dim=-1)
                .mean()
            )
            print(
                f"📊 Original hidden norm: {original_last_hidden.norm(dim=-1).mean():.4f}"
            )
            print(
                f"📊 Steered hidden norm: {steered_last_hidden.norm(dim=-1).mean():.4f}"
            )
            print(
                f"📊 Steering magnitude: {(alpha * steering_vector).norm():.4f}"
            )
            print(f"📊 Actual change magnitude: {change_magnitude:.4f}")

            if change_magnitude < 1e-6:
                print(
                    "⚠️  WARNING: Very small change detected - steering might not be effective!"
                )

        # Save steered outputs
        steered_outputs = llm.output.save()

    # Extract steered logits and decode top tokens
    steered_logits = steered_outputs[
        "logits"
    ]  # [batch_size, seq_len, vocab_size]
    steered_last_logits = steered_logits[:, -1, :]  # [batch_size, vocab_size]

    # Get top token for each input in batch
    steered_top_tokens = []
    for i, logits in enumerate(steered_last_logits):
        probs = F.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, 1)
        token = llm.tokenizer.decode(
            [top_idx.item()], skip_special_tokens=True
        )
        steered_top_tokens.append((token, top_prob.item()))

    results["steered"] = steered_top_tokens

    if debug:
        print(
            f"🎯 Steered top tokens: {steered_top_tokens[:3]}..."
        )  # Show first 3

        # Check how many changed
        changed_count = sum(
            1
            for orig, steer in zip(original_top_tokens, steered_top_tokens)
            if orig[0] != steer[0]
        )
        print(f"✅ Changed predictions: {changed_count}/{len(input_texts)}")
        print(f"{'='*60}")

    return results


def print_batch_token_comparison(
    results: dict, input_texts: list[str], concept: str
) -> None:
    """Pretty print the batch token comparison results in a two-column format."""
    print(f"\n{'='*100}")
    print(f"BATCH STEERING COMPARISON FOR CONCEPT: {concept}")
    print(f"{'='*100}")
    print(f"{'Input Text':<50} {'Original → Steered':<30} {'Changed':<10}")
    print(f"{'-'*100}")

    original_tokens = results["original"]
    steered_tokens = results["steered"]

    for i, input_text in enumerate(input_texts):
        # Truncate long input texts for display
        display_text = (
            input_text[:47] + "..." if len(input_text) > 47 else input_text
        )

        orig_token, orig_prob = original_tokens[i]
        steer_token, steer_prob = steered_tokens[i]

        # Format token comparison
        comparison = f"{orig_token} → {steer_token}"
        changed = "✓" if orig_token != steer_token else "✗"

        print(f"{display_text:<50} {comparison:<30} {changed:<10}")

    # Summary statistics
    changed_count = sum(
        1
        for orig, steer in zip(original_tokens, steered_tokens)
        if orig[0] != steer[0]
    )
    print(f"{'-'*100}")
    print(
        f"SUMMARY: {changed_count}/{len(input_texts)} predictions changed ({changed_count/len(input_texts)*100:.1f}%)"
    )
    print(f"{'='*100}\n")


def evaluate_steering_on_prompts(
    steering_vector: torch.Tensor,
    concept: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    test_prompts: list = None,
) -> None:
    """
    Evaluate steering vector on a set of test prompts using vectorized batch processing.
    """
    if test_prompts is None:
        test_prompts = [
            "Long live the",
            "The lion is the",
            "In the hierarchy of medieval society, the highest rank was the",
            "Arthur was a legendary",
            "He was known as the warrior",
            "In a monarchy, the ruler is usually a",
            "He sat on the throne, the",
            "A sovereign ruler in a monarchy is often a",
            "His domain was vast, for he was a",
            "The lion, in many cultures, is considered the",
            "He wore a crown, signifying he was the",
            "A male sovereign who reigns over a kingdom is a",
            "Every kingdom has its ruler, typically a",
            "The prince matured and eventually became the",
            "In the deck of cards, alongside the queen is the",
        ]

    try:
        print(
            f"Evaluating steering vector for concept '{concept}' on {len(test_prompts)} prompts..."
        )
        print(f"Using vectorized batch processing with nnsight...")

        # Use the new vectorized batch function
        results = compare_top_tokens_with_steering_batch(
            model_name=model_name,
            input_texts=test_prompts,
            steering_vector=steering_vector,
            layer_idx=16,  # Use middle layer
            alpha=5.0,  # Steering strength
            debug=True,  # Enable debug output
        )

        # Print results in a nice batch format
        print_batch_token_comparison(results, test_prompts, concept)

    except Exception as e:
        print(f"Error running evaluation: {e}")
        print("Skipping steering evaluation - check your setup and API key")
        import traceback

        traceback.print_exc()
        return


def main(args):
    tilde_z_test, z_test = utils.load_test_data(
        datafile=args.datafile,
    )
    z_test = utils.tensorify(z_test, device)
    print(z_test.shape)
    tilde_z_test = utils.tensorify(tilde_z_test, device)
    with open(args.dataconfig, "r") as file:
        dataconfig = Box(yaml.safe_load(file))
    concept_labels_test_path = os.path.dirname(os.path.abspath(args.datafile))
    concept_labels_test = utils.load_json(
        os.path.join(concept_labels_test_path, "concept_labels_test.json")
    )
    concept_test_sets = split_test_data_by_concept(
        tilde_z_test=tilde_z_test,
        z_test=z_test,
        concept_labels_test=concept_labels_test,
    )
    # contains keys for each concept, with values as tuples of tensors
    # (tilde_z_test, z_test)
    if args.modeltype == "llamascope":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = (
            load_llamascope_checkpoint()
        )
        decoder_weight_matrices = [decoder_weight]
        decoder_bias_vectors = [decoder_bias]
        encoder_weight_matrices = [encoder_weight]
        encoder_bias_vectors = [encoder_bias]
        concept_metrics = {}
        for concept, concept_test_set in concept_test_sets.items():
            tilde_z_test, z_test = concept_test_set
            # Compute cosine similarities for each concept
            mean_cos, std_cos, steering_vector = (
                get_max_cos_and_steering_vector_for_concept(
                    z_test,
                    tilde_z_test,
                    decoder_weight,
                    decoder_bias,
                )
            )
            print(f"Mean Cosine Similarity for concept {concept}: {mean_cos}")
            print(f"Std Cosine Similarity for concept {concept}:  {std_cos}")
            concept_metrics[concept] = {
                "mean_cos": mean_cos,
                "std_cos": std_cos,
                "steering_vector": steering_vector,
            }

            # Evaluate steering vector on test prompts
            if (
                steering_vector is not None
                and hasattr(args, "evaluate_steering")
                and args.evaluate_steering
            ):
                evaluate_steering_on_prompts(
                    steering_vector=steering_vector, concept=concept
                )

            if args.ood_data:
                dirname = os.path.dirname(os.path.abspath(args.ood_data))
                ood_concept_id = utils.load_json(
                    os.path.join(dirname, "concept_labels_test.json")
                )[0]
                if concept == ood_concept_id:
                    get_ood_cosine_similarity(
                        steering_vector, args.ood_data, decoder_bias
                    )
    elif args.modeltype == "ssae":
        # Load the decoder weights from the LinearSAE model
        # (assuming the model is already trained)
        modeldirs = args.modeldirs
        if len(modeldirs) < 2:
            raise ValueError(
                "You must provide at least two model directories."
            )
        print(f"Loading model configs from {len(modeldirs)} directories...")
        model_configs = [load_model_config(modeldir) for modeldir in modeldirs]
        check_model_config_consistency(model_configs)
        print("Loading decoder weight matrices...")
        decoder_weight_matrices = [
            load_ssae(modeldir, dataconfig)[0] for modeldir in modeldirs
        ]
        decoder_bias_vectors = [
            load_ssae(modeldir, dataconfig)[1] for modeldir in modeldirs
        ]
        encoder_weight_matrices = [
            load_ssae(modeldir, dataconfig)[2] for modeldir in modeldirs
        ]
        encoder_bias_vectors = [
            load_ssae(modeldir, dataconfig)[3] for modeldir in modeldirs
        ]
        print("Computing pairwise MCCs...")
        mccs = compute_all_pairwise_mccs(decoder_weight_matrices)

        mean_mcc = np.mean(mccs)
        std_mcc = np.std(mccs)

        print("\nPairwise MCCs:")
        for i, (a, b) in enumerate(
            itertools.combinations(range(len(modeldirs)), 2)
        ):
            print(f"Model {a+1} vs Model {b+1}: MCC = {mccs[i]:.4f}")
        print(f"\nMean MCC: {mean_mcc:.4f}")
        print(f"Std  MCC: {std_mcc:.4f}")
        concept_metrics = {}
        for concept, concept_test_set in concept_test_sets.items():
            tilde_z_test, z_test = concept_test_set
            # Compute cosine similarities for each concept
            mean_cos, std_cos, steering_vector = (
                get_max_cos_and_steering_vector_for_concept(
                    z_test,
                    tilde_z_test,
                    decoder_weight_matrices[0],
                    decoder_bias_vectors[0],
                )
            )
            print(f"Mean Cosine Similarity for concept {concept}: {mean_cos}")
            print(f"Std Cosine Similarity for concept {concept}:  {std_cos}")
            concept_metrics[concept] = {
                "mean_cos": mean_cos,
                "std_cos": std_cos,
                "steering_vector": steering_vector,
            }

            # Evaluate steering vector on test prompts
            if (
                steering_vector is not None
                and hasattr(args, "evaluate_steering")
                and args.evaluate_steering
            ):
                evaluate_steering_on_prompts(
                    steering_vector=steering_vector, concept=concept
                )

            if args.ood_data:
                dirname = os.path.dirname(os.path.abspath(args.ood_data))
                ood_concept_id = utils.load_json(
                    os.path.join(dirname, "concept_labels_test.json")
                )[0]
                if concept == ood_concept_id:
                    get_ood_cosine_similarity(
                        steering_vector, args.ood_data, decoder_bias_vectors[0]
                    )
    elif args.modeltype == "pca":
        cosines_pca = take_pca(z_test, tilde_z_test)
        print(
            "USING PCA cosine similarities",
            np.mean(cosines_pca),
            np.std(cosines_pca),
        )
    else:
        raise ValueError
    mean_over_max_logits, entropy = get_concept_detection_logits(
        z_test,
        encoder_weight_matrices[0],
        encoder_bias_vectors[0],
        decoder_bias_vectors[0],
        args,
    )
    print("Concept Detection Scores", mean_over_max_logits, entropy)
    print(f"Using {args.modeltype}, computing max cosine similarities for ")
    print("...", mean_cos, std_cos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile")
    parser.add_argument("dataconfig")
    parser.add_argument(
        "--modeltype", default="ssae", choices=["llamascope", "ssae"]
    )
    parser.add_argument(
        "--modeldirs",
        nargs="+",
        type=str,
        help="List of model directories to compare.",
    )
    parser.add_argument(
        "--ood-data", help="Path to OOD data file using the same dataconfig."
    )
    parser.add_argument(
        "--evaluate-steering",
        action="store_true",
        help="Evaluate steering vectors on test prompts with Llama3 model",
    )
    args = parser.parse_args()
    with dbg.debug_on_exception():
        main(args)
