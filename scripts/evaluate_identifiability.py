#!/usr/bin/env python3
"""
Evaluate identifiability metrics (MCC or DCI) across datasets.

Compares:
- Custom SSAE (DictLinearAE)
- Trained SAE (ssae/sae.py SAE class) — optional
- Pretrained SAE (gemmascope/pythia/llama) — optional

Metrics:
- MCC: max Pearson correlation per feature + steering cosine similarity
- DCI: Disentanglement, Completeness, Informativeness (GBT-based)
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssae import DictLinearAE
from loaders import (
    load_llamascope_checkpoint,
    load_gemmascope_checkpoint,
    load_pythia_sae_checkpoint,
)
from utils.data_utils import load_dataset as load_eval_dataset
from utils.data_utils import generate_counterfactual_pairs
from utils.dci import compute_dci

SUPPORTED_DATASETS = [
    "truthful-qa",
    "sycophancy",
    "refusal",
    "safearena",
    "wildjailbreak",
    "bias-in-bios",
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "masc-fem-mixed",
    "2-binary",
    "corr-binary",
]

COUNTERFACTUAL_DATASETS = [
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "masc-fem-mixed",
    "2-binary",
    "corr-binary",
]


# ==============================================================================
# ADAPTER CLASSES — unified interface for different SAE types
# ==============================================================================


class _TrainedSAEAdapter:
    """Wraps ssae/sae.py SAE to match DictLinearAE interface for evaluation.

    SAE uses:
    - self.Ad: decoder weight [dimin, width] — same layout as DictLinearAE decoder.weight
    - forward(x, return_hidden=True) returns (x_hat, codes)
    """

    def __init__(self, sae):
        self._sae = sae
        # Expose Ad as decoder.weight for steering vector extraction
        self.decoder = type("Decoder", (), {"weight": sae.Ad})()

    def __call__(self, x):
        return self._sae(x, return_hidden=True)

    def eval(self):
        self._sae.eval()
        return self


class _PretrainedSAEAdapter:
    """Wraps pretrained SAE weights (gemmascope/pythia/llama) with ReLU encoder.

    Weight conventions (after loader transpose):
    - encoder_weight: [hid, rep_dim]
    - decoder_weight: [rep_dim, hid]
    """

    def __init__(self, decoder_weight, decoder_bias, encoder_weight, encoder_bias):
        # Cast to float32 (gemmascope may be bfloat16)
        self._decoder_weight = decoder_weight.float()
        self._decoder_bias = decoder_bias.float()
        self._encoder_weight = encoder_weight.float()
        self._encoder_bias = encoder_bias.float()
        # Expose decoder.weight for steering vector extraction
        self.decoder = type("Decoder", (), {"weight": self._decoder_weight})()

    def __call__(self, x):
        x = x.float()
        h = F.relu(x @ self._encoder_weight.T + self._encoder_bias)
        x_hat = h @ self._decoder_weight.T + self._decoder_bias
        return x_hat, h

    def eval(self):
        return self


# ==============================================================================
# MODEL LOADING FUNCTIONS
# ==============================================================================


def load_sae_model(model_path: Path):
    """Load custom SSAE model (DictLinearAE)."""
    import yaml

    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weights_path = model_path / "weights.pth"
    state_dict = torch.load(weights_path, map_location="cpu")

    rep_dim = state_dict["encoder.weight"].shape[1]
    hid_dim = state_dict["encoder.weight"].shape[0]

    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_trained_sae(model_path: Path):
    """Load a trained SAE model (from ssae/sae.py) and wrap it."""
    import yaml
    from ssae.sae import SAE

    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weights_path = model_path / "weights.pth"
    state_dict = torch.load(weights_path, map_location="cpu")

    # Infer dimensions from state_dict
    width = state_dict["Ae"].shape[0]
    dimin = state_dict["Ae"].shape[1]
    sae_type = cfg.get("sae_type", "relu")
    kval_topk = cfg.get("kval_topk", None)
    mp_kval = cfg.get("mp_kval", None)

    model = SAE(
        dimin=dimin,
        width=width,
        sae_type=sae_type,
        kval_topk=kval_topk,
        mp_kval=mp_kval,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return _TrainedSAEAdapter(model)


def load_pretrained_sae(embedding_model: str, layer: int = None, hf_token: str = None):
    """Load pretrained SAE (gemmascope/pythia/llama) and wrap it."""
    if embedding_model == "pythia":
        layer = layer or 5
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = (
            load_pythia_sae_checkpoint(layer, hf_token)
        )
    elif embedding_model == "gemma":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = (
            load_gemmascope_checkpoint()
        )
    elif embedding_model == "llama":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = (
            load_llamascope_checkpoint()
        )
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    print(f"Pretrained SAE dimensions: encoder={encoder_weight.shape}, decoder={decoder_weight.shape}")

    return _PretrainedSAEAdapter(decoder_weight, decoder_bias, encoder_weight, encoder_bias)


# ==============================================================================
# DATA UTILITIES
# ==============================================================================


def flatten_pairs(pairs):
    """Flatten contrastive pairs into individual texts with binary labels.

    For each pair [text_a, text_b], text_a gets label 0 and text_b gets label 1.
    """
    texts = []
    labels = []
    for text_a, text_b in pairs:
        texts.append(text_a)
        labels.append(0)
        texts.append(text_b)
        labels.append(1)
    return texts, labels


def extract_embeddings(texts, embedding_model: str):
    """Extract embeddings from texts using specified model."""
    if embedding_model == "pythia":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "EleutherAI/pythia-70m-deduped"
        layer = 5
        batch_size = 32
    elif embedding_model == "gemma":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "google/gemma-2-2b-it"
        layer = 25
        batch_size = 16
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    print(f"Loading {embedding_model} model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    print(f"Extracting embeddings for {len(texts)} samples...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]

            batch_embeddings = []
            for j, input_ids in enumerate(inputs["input_ids"]):
                non_pad_indices = (
                    input_ids != tokenizer.pad_token_id
                ).nonzero(as_tuple=True)[0]
                if len(non_pad_indices) > 0:
                    last_token_idx = non_pad_indices[-1]
                    embedding = hidden_states[j, last_token_idx]
                else:
                    embedding = hidden_states[j, 0]
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

    return torch.stack(embeddings)


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================


def get_sae_activations(model, embeddings):
    """Get SAE activations from embeddings."""
    activations = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            _, acts = model(batch)
            activations.append(acts.cpu())

    return torch.cat(activations, dim=0)


def compute_linear_probe_mcc(activations, labels, seed=42):
    """Train a linear probe and return Pearson correlation on held-out test set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = activations.float().numpy()
    y = np.array(labels, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test).astype(np.float32)
    accuracy = (y_pred == y_test).mean()

    pred_centered = y_pred - y_pred.mean()
    label_centered = y_test - y_test.mean()
    pred_norm = np.linalg.norm(pred_centered)
    label_norm = np.linalg.norm(label_centered)
    if pred_norm == 0 or label_norm == 0:
        corr = 0.0
    else:
        corr = float(pred_centered @ label_centered / (pred_norm * label_norm))

    return {"mcc": corr, "accuracy": accuracy}


def compute_max_correlation_mcc(activations, labels):
    """Find max correlation feature and return Pearson correlation as MCC score."""
    activations = activations.float()
    labels = torch.tensor(labels, dtype=torch.float32)

    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)

    labels_centered = labels - labels.mean()
    labels_std = labels_centered.norm()

    numerator = acts_centered.T @ labels_centered
    denominator = acts_std.squeeze() * labels_std

    mask = denominator != 0
    correlations = torch.zeros(activations.shape[1])
    correlations[mask] = numerator[mask] / denominator[mask]

    best_feature_idx = correlations.argmax().item()
    best_correlation = correlations[best_feature_idx].item()

    return {
        "mcc": best_correlation,
        "feature_idx": best_feature_idx,
        "correlation": best_correlation,
        "max_correlation": correlations.max().item(),
    }


def compute_steering_cosine_sim(embeddings, sae_model, feature_idx):
    """Compute cosine similarity between pair shift vectors and the SAE steering vector.

    The steering vector is the decoder column for the given feature.
    Shift vectors are (z_tilde - z) for each test pair.
    Embeddings are in flattened pair order: [z_0, z_tilde_0, z_1, z_tilde_1, ...].
    """
    steering = sae_model.decoder.weight[:, feature_idx].detach().float()

    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    shifts = z_tilde - z

    cos_sims = F.cosine_similarity(shifts, steering.unsqueeze(0), dim=1)

    return {
        "mean_cos_sim": cos_sims.mean().item(),
        "std_cos_sim": cos_sims.std().item(),
        "median_cos_sim": cos_sims.median().item(),
    }


def compute_decoder_projection_mcc(model, embeddings, labels):
    """Compute MCC by projecting embeddings directly onto decoder columns.

    Instead of running through the full encoder (which may have distribution
    mismatch for SSAEs trained on differences), this directly measures how
    well each learned dictionary direction separates the two classes.

    For all SAE types, decoder.weight has shape (rep_dim, hid).
    Column j is the j-th concept direction in embedding space.

    Args:
        model: Any SAE model with model.decoder.weight attribute.
        embeddings: Individual embeddings, shape (N, rep_dim).
        labels: Binary labels, length N.

    Returns:
        Dict with mcc, feature_idx, and steering cosine similarity.
    """
    embeddings_f = embeddings.float()
    labels_t = torch.tensor(labels, dtype=torch.float32)

    # decoder.weight: (rep_dim, hid) — each column is a concept direction
    W = model.decoder.weight.detach().float()  # (rep_dim, hid)

    # Project each embedding onto each decoder column
    projections = embeddings_f @ W  # (N, hid)

    # Pearson correlation between each column's projections and labels
    proj_centered = projections - projections.mean(dim=0, keepdim=True)
    proj_std = proj_centered.norm(dim=0, keepdim=True)

    labels_centered = labels_t - labels_t.mean()
    labels_std = labels_centered.norm()

    numerator = proj_centered.T @ labels_centered  # (hid,)
    denominator = proj_std.squeeze() * labels_std

    mask = denominator != 0
    correlations = torch.zeros(projections.shape[1])
    correlations[mask] = numerator[mask] / denominator[mask]

    best_feature_idx = correlations.argmax().item()
    best_correlation = correlations[best_feature_idx].item()

    # Steering cos sim for the best decoder-projection feature
    steering = W[:, best_feature_idx]
    z = embeddings_f[0::2]
    z_tilde = embeddings_f[1::2]
    shifts = z_tilde - z
    cos_sims = F.cosine_similarity(shifts, steering.unsqueeze(0), dim=1)

    return {
        "mcc": best_correlation,
        "feature_idx": best_feature_idx,
        "steer": {
            "mean_cos_sim": cos_sims.mean().item(),
            "std_cos_sim": cos_sims.std().item(),
        },
    }


def evaluate_sae(name, model, embeddings, activations, labels, run_linear_probe=False):
    """Run full evaluation (MCC + steering + optional probe) for a single SAE."""
    results = compute_max_correlation_mcc(activations, labels)
    steer = compute_steering_cosine_sim(embeddings, model, results["feature_idx"])

    print(f"  MCC:            {results['mcc']:.4f} (feature {results['feature_idx']})")
    print(f"  Steering cos:   {steer['mean_cos_sim']:.4f} (+/- {steer['std_cos_sim']:.4f})")

    if run_linear_probe:
        probe = compute_linear_probe_mcc(activations, labels)
        print(f"  Linear probe:   corr={probe['mcc']:.4f}  acc={probe['accuracy']:.4f}")
        print(f"  Gap (probe-MCC): {probe['mcc'] - results['mcc']:.4f}")
        results["probe"] = probe

    results["steer"] = steer
    return results


def evaluate_sae_dci(name, activations, factors, seed=42):
    """Run DCI evaluation for a single SAE.

    Args:
        name: Display name for the model.
        activations: SAE codes, shape (n_samples, n_codes).
        factors: Ground truth factors, shape (n_samples,) or (n_samples, n_factors).
    """
    codes_np = activations.float().numpy()
    factors_np = np.array(factors, dtype=np.float64)

    results = compute_dci(
        codes_np, factors_np, train_fraction=0.8, random_state=seed,
    )

    print(f"  Disentanglement:        {results['disentanglement']:.4f}")
    print(f"  Completeness:           {results['completeness']:.4f}")
    print(f"  Informativeness train:  {results['informativeness_train']:.4f}")
    print(f"  Informativeness test:   {results['informativeness_test']:.4f}")

    return results


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate identifiability metrics (MCC or DCI) across datasets"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to SSAE model directory"
    )
    parser.add_argument(
        "--metric",
        choices=["mcc", "dci"],
        default="mcc",
        help="Metric to compute: mcc (max correlation) or dci (disentanglement/completeness/informativeness)",
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="bias-in-bios",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["pythia", "gemma", "llama"],
        default="pythia",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--trained-sae",
        type=Path,
        default=None,
        help="Path to trained SAE model directory (ssae/sae.py SAE class)",
    )
    parser.add_argument(
        "--pretrained-sae",
        action="store_true",
        help="Compare with pretrained SAE (gemmascope/pythia/llama)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for dataset loading",
    )
    parser.add_argument(
        "--layer", type=int, help="Layer for pretrained SAE (pythia only)"
    )
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="Run linear probe baseline on SAE activations and raw embeddings",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to h5 file with pre-computed embeddings (loads cfc_test, skips LLM extraction)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------
    concept_labels = None

    if args.embeddings:
        # Load pre-computed test embeddings from h5 file (cfc_test)
        print(f"Loading pre-computed test embeddings from {args.embeddings}...")
        with h5py.File(args.embeddings, "r") as f:
            cfc_test = f["cfc_test"][:]  # (N, 2, rep_dim)

        n_pairs = cfc_test.shape[0]

        # Interleave: [z_0, z_tilde_0, z_1, z_tilde_1, ...]
        embeddings = torch.zeros(2 * n_pairs, cfc_test.shape[-1])
        embeddings[0::2] = torch.from_numpy(cfc_test[:, 0])
        embeddings[1::2] = torch.from_numpy(cfc_test[:, 1])
        labels = [0, 1] * n_pairs

        print(f"Loaded {n_pairs} test pairs -> {len(embeddings)} embeddings, "
              f"shape {embeddings.shape}")
    else:
        # Fall back to loading text + re-extracting embeddings via LLM
        print(f"Loading dataset: {args.dataset}...")
        if args.dataset in COUNTERFACTUAL_DATASETS:
            split_index = int(0.9 * args.num_samples)
            _, cfc_test_tuples, concept_labels = generate_counterfactual_pairs(
                args.dataset, split_index, args.num_samples
            )
        else:
            _, cfc_test_tuples = load_eval_dataset(
                args.dataset, num_samples=args.num_samples
            )

        texts, labels = flatten_pairs(cfc_test_tuples)

        n_class0 = sum(1 for l in labels if l == 0)
        n_class1 = sum(1 for l in labels if l == 1)
        print(f"Loaded {len(texts)} texts ({n_class0} class-0, {n_class1} class-1) "
              f"from {len(cfc_test_tuples)} test pairs")

        embeddings = extract_embeddings(texts, args.embedding_model)

    print(f"Embeddings shape: {embeddings.shape}")

    # -------------------------------------------------------------------------
    # Load all SAE models
    # -------------------------------------------------------------------------
    print("\nLoading SAE models...")

    # Custom SSAE (required)
    ssae_model = load_sae_model(args.model_path)
    ssae_activations = get_sae_activations(ssae_model, embeddings)
    print(f"  SSAE loaded: {ssae_activations.shape[1]} features")

    # Trained SAE (optional)
    trained_sae_model = None
    trained_sae_activations = None
    if args.trained_sae:
        trained_sae_model = load_trained_sae(args.trained_sae)
        trained_sae_activations = get_sae_activations(trained_sae_model, embeddings)
        print(f"  Trained SAE loaded: {trained_sae_activations.shape[1]} features")

    # Pretrained SAE (optional)
    pretrained_sae_model = None
    pretrained_sae_activations = None
    if args.pretrained_sae:
        try:
            pretrained_sae_model = load_pretrained_sae(
                args.embedding_model, args.layer, args.hf_token
            )
            pretrained_sae_activations = get_sae_activations(pretrained_sae_model, embeddings)
            print(f"  Pretrained SAE loaded: {pretrained_sae_activations.shape[1]} features")
        except Exception as e:
            print(f"  Pretrained SAE Error: {e}")

    # -------------------------------------------------------------------------
    # Determine concept groups for evaluation
    # -------------------------------------------------------------------------
    if concept_labels is not None:
        concepts = sorted(set(concept_labels))
    else:
        concepts = [None]

    # -------------------------------------------------------------------------
    # Evaluate per concept
    # -------------------------------------------------------------------------
    metric = args.metric
    print(f"\nMetric: {metric.upper()}")

    for concept in concepts:
        if concept is not None:
            pair_indices = [i for i, c in enumerate(concept_labels) if c == concept]
            flat_indices = []
            for i in pair_indices:
                flat_indices.extend([2 * i, 2 * i + 1])
            c_embeddings = embeddings[flat_indices]
            c_ssae_acts = ssae_activations[flat_indices]
            c_labels = [labels[j] for j in flat_indices]
            concept_name = concept
        else:
            c_embeddings = embeddings
            c_ssae_acts = ssae_activations
            c_labels = labels
            flat_indices = None
            concept_name = args.dataset

        n_pairs = len(c_labels) // 2

        # --- SSAE ---
        print(f"\n{'='*60}")
        print(f"SSAE — {concept_name} ({n_pairs} pairs)")
        print(f"{'='*60}")
        if metric == "mcc":
            ssae_results = evaluate_sae(
                "SSAE", ssae_model, c_embeddings, c_ssae_acts, c_labels, args.linear_probe
            )
            ssae_dec_results = compute_decoder_projection_mcc(
                ssae_model, c_embeddings, c_labels
            )
            print(f"  Decoder MCC:    {ssae_dec_results['mcc']:.4f} (feature {ssae_dec_results['feature_idx']})")
            print(f"  Decoder steer:  {ssae_dec_results['steer']['mean_cos_sim']:.4f} (+/- {ssae_dec_results['steer']['std_cos_sim']:.4f})")
        else:
            ssae_results = evaluate_sae_dci("SSAE", c_ssae_acts, c_labels)

        # --- Trained SAE ---
        if trained_sae_model is not None:
            c_trained_acts = (
                trained_sae_activations[flat_indices]
                if flat_indices is not None
                else trained_sae_activations
            )

            print(f"\n{'-'*60}")
            print(f"Trained SAE — {concept_name}")
            print(f"{'-'*60}")
            if metric == "mcc":
                trained_results = evaluate_sae(
                    "Trained SAE", trained_sae_model, c_embeddings, c_trained_acts, c_labels, args.linear_probe
                )
                trained_dec_results = compute_decoder_projection_mcc(
                    trained_sae_model, c_embeddings, c_labels
                )
                print(f"  Decoder MCC:    {trained_dec_results['mcc']:.4f} (feature {trained_dec_results['feature_idx']})")
                print(f"  Decoder steer:  {trained_dec_results['steer']['mean_cos_sim']:.4f} (+/- {trained_dec_results['steer']['std_cos_sim']:.4f})")
                print(f"\n  vs SSAE:")
                print(f"    MCC diff:     {ssae_results['mcc'] - trained_results['mcc']:.4f} (SSAE - Trained)")
                print(f"    Steer diff:   {ssae_results['steer']['mean_cos_sim'] - trained_results['steer']['mean_cos_sim']:.4f}")
                print(f"    Dec MCC diff: {ssae_dec_results['mcc'] - trained_dec_results['mcc']:.4f} (SSAE - Trained)")
            else:
                trained_results = evaluate_sae_dci("Trained SAE", c_trained_acts, c_labels)
                print(f"\n  vs SSAE:")
                print(f"    Disent diff:  {ssae_results['disentanglement'] - trained_results['disentanglement']:.4f} (SSAE - Trained)")
                print(f"    Compl diff:   {ssae_results['completeness'] - trained_results['completeness']:.4f}")

        # --- Pretrained SAE ---
        if pretrained_sae_model is not None:
            c_pretrained_acts = (
                pretrained_sae_activations[flat_indices]
                if flat_indices is not None
                else pretrained_sae_activations
            )

            print(f"\n{'-'*60}")
            print(f"Pretrained SAE ({args.embedding_model}) — {concept_name}")
            print(f"{'-'*60}")
            if metric == "mcc":
                pretrained_results = evaluate_sae(
                    "Pretrained SAE", pretrained_sae_model, c_embeddings, c_pretrained_acts, c_labels, args.linear_probe
                )
                pretrained_dec_results = compute_decoder_projection_mcc(
                    pretrained_sae_model, c_embeddings, c_labels
                )
                print(f"  Decoder MCC:    {pretrained_dec_results['mcc']:.4f} (feature {pretrained_dec_results['feature_idx']})")
                print(f"  Decoder steer:  {pretrained_dec_results['steer']['mean_cos_sim']:.4f} (+/- {pretrained_dec_results['steer']['std_cos_sim']:.4f})")
                print(f"\n  vs SSAE:")
                print(f"    MCC diff:     {ssae_results['mcc'] - pretrained_results['mcc']:.4f} (SSAE - Pretrained)")
                print(f"    Steer diff:   {ssae_results['steer']['mean_cos_sim'] - pretrained_results['steer']['mean_cos_sim']:.4f}")
                print(f"    Dec MCC diff: {ssae_dec_results['mcc'] - pretrained_dec_results['mcc']:.4f} (SSAE - Pretrained)")
            else:
                pretrained_results = evaluate_sae_dci("Pretrained SAE", c_pretrained_acts, c_labels)
                print(f"\n  vs SSAE:")
                print(f"    Disent diff:  {ssae_results['disentanglement'] - pretrained_results['disentanglement']:.4f} (SSAE - Pretrained)")
                print(f"    Compl diff:   {ssae_results['completeness'] - pretrained_results['completeness']:.4f}")

        # --- Raw embeddings linear probe ---
        if args.linear_probe and metric == "mcc":
            print(f"\n{'-'*60}")
            print(f"Raw Embeddings Linear Probe — {concept_name}")
            print(f"{'-'*60}")
            probe_emb = compute_linear_probe_mcc(c_embeddings, c_labels)
            print(f"  Pearson corr: {probe_emb['mcc']:.4f}")
            print(f"  Accuracy:     {probe_emb['accuracy']:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
