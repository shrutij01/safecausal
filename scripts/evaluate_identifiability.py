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

import re
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import argparse
import sys
import os
from pathlib import Path
from collections import Counter

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


def _get_decoder_bias(model):
    """Extract decoder bias from various SAE model types.

    Returns a (rep_dim,) float tensor, or None if unavailable.
    """
    # DictLinearAE: model.decoder is nn.Linear with .bias
    if hasattr(model, "decoder") and hasattr(model.decoder, "bias"):
        b = model.decoder.bias
        if b is not None:
            return b.detach().float()
    # _PretrainedSAEAdapter: stores _decoder_bias
    if hasattr(model, "_decoder_bias"):
        return model._decoder_bias.detach().float()
    # _TrainedSAEAdapter: wraps SAE with bd parameter (shape 1, rep_dim)
    if hasattr(model, "_sae") and hasattr(model._sae, "bd"):
        return model._sae.bd.detach().float().squeeze(0)
    return None


def resolve_model_path(path: Path, root_dir: Path = None) -> Path:
    """Resolve a model path: if not absolute, join with root_dir."""
    if path.is_absolute():
        return path
    if root_dir is None:
        raise ValueError(
            f"Path '{path}' is relative but --root-dir was not provided. "
            "Either use an absolute path or supply --root-dir."
        )
    return root_dir / path


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


def compute_steering_cosine_sim(embeddings, sae_model):
    """Find the decoder column that best steers z toward z_tilde.

    For each pair (z, z_tilde) and each decoder column d_j:
        z_hat = normalize(z + d_j + bias)
        cos  = cos_sim(z_hat, normalize(z_tilde))
    Takes max over decoder columns per pair.
    The most frequently selected column is reported as the steering vector.

    Embeddings are in flattened pair order: [z_0, z_tilde_0, z_1, z_tilde_1, ...].
    """
    z = embeddings[0::2].float()           # (N, D)
    z_tilde = embeddings[1::2].float()     # (N, D)
    N, D = z.shape

    # Decoder columns: (D, V) where V = number of features
    W = sae_model.decoder.weight.detach().float()  # (D, V)
    V = W.shape[1]

    z_tilde_norm = F.normalize(z_tilde, dim=-1)  # (N, D)

    # Include decoder bias if available
    bias = _get_decoder_bias(sae_model)  # (D,) or None

    # Track running max cosine per pair and corresponding column index
    max_cos = torch.full((N,), -2.0)
    best_col = torch.zeros(N, dtype=torch.long)

    # Batch over decoder columns to limit memory (~1 GB target)
    batch_v = max(1, int(1e9 / (N * D * 4)))
    batch_v = min(batch_v, V)

    for start in range(0, V, batch_v):
        end = min(start + batch_v, V)
        d_batch = W[:, start:end].T  # (batch_v, D)

        # z_hat = z + d_j [+ bias]
        z_hat = z.unsqueeze(1) + d_batch.unsqueeze(0)  # (N, batch_v, D)
        if bias is not None:
            z_hat = z_hat + bias.unsqueeze(0).unsqueeze(0)  # broadcast (1, 1, D)

        z_hat = F.normalize(z_hat, dim=-1)  # (N, batch_v, D)

        # Cosine similarity with z_tilde
        cos = (z_hat * z_tilde_norm.unsqueeze(1)).sum(dim=-1)  # (N, batch_v)

        # Update running max
        batch_max, batch_argmax = cos.max(dim=1)  # (N,)
        improved = batch_max > max_cos
        max_cos[improved] = batch_max[improved]
        best_col[improved] = batch_argmax[improved] + start

    # Most frequently selected column = steering vector
    col_counts = Counter(best_col.tolist())
    steering_col_idx, steering_col_count = col_counts.most_common(1)[0]

    return {
        "mean_cos_sim": max_cos.mean().item(),
        "std_cos_sim": max_cos.std().item(),
        "median_cos_sim": max_cos.median().item(),
        "steering_col_idx": steering_col_idx,
        "steering_col_freq": steering_col_count / N,
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
        Dict with mcc and feature_idx.
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

    return {
        "mcc": best_correlation,
        "feature_idx": best_feature_idx,
    }


def evaluate_sae(name, model, embeddings, activations, labels, run_linear_probe=False):
    """Run full evaluation (MCC + steering + optional probe) for a single SAE."""
    results = compute_max_correlation_mcc(activations, labels)
    steer = compute_steering_cosine_sim(embeddings, model)

    print(f"  MCC:            {results['mcc']:.4f} (feature {results['feature_idx']})")
    print(f"  Steering cos:   {steer['mean_cos_sim']:.4f} (+/- {steer['std_cos_sim']:.4f})"
          f"  [col {steer['steering_col_idx']}, freq {steer['steering_col_freq']:.0%}]")

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
        nargs="+",
        default=None,
        help="Path(s) to trained SAE model directories. "
             "Resolved relative to --root-dir if not absolute.",
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
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Common root directory (e.g., .../run_out/). "
             "model_path and --trained-sae are resolved relative to this if not absolute.",
    )
    parser.add_argument(
        "--all-sae-types",
        action="store_true",
        help="Auto-discover all SAE variants (relu, topk, jumprelu) matching the SSAE's "
             "dataset/model/seed from the same root directory.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Find all seed variants for SSAE and SAE models and report mean +/- std. "
             "Requires --all-sae-types or explicit model paths.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Resolve paths relative to root-dir
    # -------------------------------------------------------------------------
    root_dir = args.root_dir

    # Resolve model_path (SSAE)
    args.model_path = resolve_model_path(args.model_path, root_dir)

    # Resolve trained-sae paths
    if args.trained_sae:
        args.trained_sae = [resolve_model_path(p, root_dir) for p in args.trained_sae]

    # -------------------------------------------------------------------------
    # Auto-discover SAE variants if --all-sae-types or --aggregate-seeds
    # -------------------------------------------------------------------------
    def _extract_base_name(dir_name):
        """Extract base name by stripping _seed{N} suffix."""
        m = re.match(r"(.*)_seed\d+$", dir_name)
        return m.group(1) if m else dir_name

    ssae_paths = [args.model_path]  # May expand to multiple if --aggregate-seeds

    if args.all_sae_types or args.aggregate_seeds:
        import yaml as _yaml
        from ssae.ssae import _extract_dataset_name

        # Infer root_dir from model_path if not explicitly given
        if root_dir is None:
            root_dir = args.model_path.parent  # parent of SSAE folder = run_out/

        # Read SSAE config to extract dataset, model, seed (for SAE discovery)
        ssae_cfg_path = args.model_path / "cfg.yaml"
        with open(ssae_cfg_path, "r") as f:
            ssae_cfg = _yaml.safe_load(f)

        # Extract naming components for SAE discovery
        emb_stem = Path(ssae_cfg["emb"]).stem
        dataset_name = _extract_dataset_name(emb_stem)
        extra = ssae_cfg.get("extra", {})
        model_name_raw = extra.get("model", "unknown")
        if "/" in model_name_raw:
            model_name_raw = model_name_raw.split("/")[-1]
        seed = ssae_cfg.get("seed", 0)

        # Discover SSAE seed variants if --aggregate-seeds
        # Use exact base name from provided path
        if args.aggregate_seeds:
            ssae_base = _extract_base_name(args.model_path.name)
            ssae_glob = f"{ssae_base}_seed*"
            ssae_discovered = sorted(root_dir.glob(ssae_glob))
            ssae_discovered = [d for d in ssae_discovered if d.is_dir() and (d / "weights.pth").exists()]
            if ssae_discovered:
                ssae_paths = ssae_discovered
                print(f"Auto-discovered {len(ssae_paths)} SSAE seed variants matching '{ssae_glob}':")
                for d in ssae_paths:
                    print(f"  {d.name}")
            else:
                print(f"Warning: --aggregate-seeds found no SSAE matches for '{ssae_glob}'")

        # Discover SAE variants if --all-sae-types
        if args.all_sae_types:
            # First, find SAEs for the reference seed
            ref_glob = f"sae_{dataset_name}_{model_name_raw}_*_seed{seed}"
            ref_discovered = sorted(root_dir.glob(ref_glob))
            ref_discovered = [d for d in ref_discovered if d.is_dir() and (d / "weights.pth").exists()]

            if args.aggregate_seeds and ref_discovered:
                # For each discovered SAE, find all its seed variants using exact base name
                all_discovered = []
                seen_bases = set()
                for ref_path in ref_discovered:
                    base = _extract_base_name(ref_path.name)
                    if base not in seen_bases:
                        seen_bases.add(base)
                        seed_variants = sorted(root_dir.glob(f"{base}_seed*"))
                        seed_variants = [d for d in seed_variants if d.is_dir() and (d / "weights.pth").exists()]
                        all_discovered.extend(seed_variants)
                discovered = sorted(set(all_discovered))
            else:
                discovered = ref_discovered

            if discovered:
                print(f"Auto-discovered {len(discovered)} SAE variants:")
                for d in discovered:
                    print(f"  {d.name}")
            else:
                print(f"Warning: --all-sae-types found no SAE matches in {root_dir}")

            # Merge with any explicitly provided --trained-sae paths (no duplicates)
            existing = set(args.trained_sae or [])
            merged = list(args.trained_sae or [])
            for d in discovered:
                if d not in existing:
                    merged.append(d)
            args.trained_sae = merged if merged else None

    # -------------------------------------------------------------------------
    # Expand --trained-sae to all seeds if --aggregate-seeds (without --all-sae-types)
    # -------------------------------------------------------------------------
    if args.aggregate_seeds and args.trained_sae and not args.all_sae_types:
        if root_dir is None:
            root_dir = args.model_path.parent

        expanded = []
        seen_bases = set()
        for sae_path in args.trained_sae:
            base = _extract_base_name(sae_path.name)
            if base not in seen_bases:
                seen_bases.add(base)
                seed_variants = sorted(root_dir.glob(f"{base}_seed*"))
                seed_variants = [d for d in seed_variants if d.is_dir() and (d / "weights.pth").exists()]
                if seed_variants:
                    expanded.extend(seed_variants)
                else:
                    expanded.append(sae_path)  # Keep original if no variants found

        if len(expanded) > len(args.trained_sae):
            print(f"Expanded --trained-sae to {len(expanded)} seed variants:")
            for d in sorted(set(expanded)):
                print(f"  {d.name}")
        args.trained_sae = sorted(set(expanded))

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

    # Custom SSAEs (may be multiple if --aggregate-seeds)
    ssae_list = []  # list of (name, model, activations)
    for ssae_path in ssae_paths:
        ssae_name = ssae_path.name
        ssae_model = load_sae_model(ssae_path)
        ssae_acts = get_sae_activations(ssae_model, embeddings)
        ssae_list.append((ssae_name, ssae_model, ssae_acts))
        print(f"  SSAE '{ssae_name}' loaded: {ssae_acts.shape[1]} features")

    # Trained SAEs (optional, may be multiple)
    trained_saes = []  # list of (name, model, activations)
    if args.trained_sae:
        for sae_path in args.trained_sae:
            sae_name = sae_path.name
            sae_model = load_trained_sae(sae_path)
            sae_acts = get_sae_activations(sae_model, embeddings)
            trained_saes.append((sae_name, sae_model, sae_acts))
            print(f"  Trained SAE '{sae_name}' loaded: {sae_acts.shape[1]} features")

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

    # Helper to extract model type from directory name for grouping
    def _get_model_type(name):
        if name.startswith("ssae_"):
            return "SSAE"
        # Extract SAE type: sae_{dataset}_{model}_{type}_...
        m = re.search(r"sae_[^_]+_[^_]+_([^_]+)_", name)
        if m:
            return m.group(1).upper()  # relu -> RELU, topk -> TOPK, etc.
        return name

    # Collect results for aggregation if --aggregate-seeds
    aggregated_results = {}  # {model_type: [mcc_values]}

    for concept in concepts:
        if concept is not None:
            pair_indices = [i for i, c in enumerate(concept_labels) if c == concept]
            flat_indices = []
            for i in pair_indices:
                flat_indices.extend([2 * i, 2 * i + 1])
            c_embeddings = embeddings[flat_indices]
            c_labels = [labels[j] for j in flat_indices]
            concept_name = concept
        else:
            c_embeddings = embeddings
            c_labels = labels
            flat_indices = None
            concept_name = args.dataset

        n_pairs = len(c_labels) // 2

        # --- SSAEs ---
        first_ssae_results = None  # For comparison with trained SAEs
        first_ssae_dec_results = None
        for ssae_name, ssae_model, ssae_full_acts in ssae_list:
            c_ssae_acts = (
                ssae_full_acts[flat_indices]
                if flat_indices is not None
                else ssae_full_acts
            )

            print(f"\n{'='*60}")
            print(f"SSAE [{ssae_name}] — {concept_name} ({n_pairs} pairs)")
            print(f"{'='*60}")
            if metric == "mcc":
                ssae_results = evaluate_sae(
                    ssae_name, ssae_model, c_embeddings, c_ssae_acts, c_labels, args.linear_probe
                )
                ssae_dec_results = compute_decoder_projection_mcc(
                    ssae_model, c_embeddings, c_labels
                )
                print(f"  Decoder MCC:    {ssae_dec_results['mcc']:.4f} (feature {ssae_dec_results['feature_idx']})")

                # Collect for aggregation
                if args.aggregate_seeds:
                    model_type = _get_model_type(ssae_name)
                    aggregated_results.setdefault(model_type, []).append(ssae_results['mcc'])

                # Keep first for comparison
                if first_ssae_results is None:
                    first_ssae_results = ssae_results
                    first_ssae_dec_results = ssae_dec_results
            else:
                ssae_results = evaluate_sae_dci(ssae_name, c_ssae_acts, c_labels)
                if first_ssae_results is None:
                    first_ssae_results = ssae_results

        # --- Trained SAEs ---
        for sae_name, sae_model, sae_full_acts in trained_saes:
            c_trained_acts = (
                sae_full_acts[flat_indices]
                if flat_indices is not None
                else sae_full_acts
            )

            print(f"\n{'-'*60}")
            print(f"Trained SAE [{sae_name}] — {concept_name}")
            print(f"{'-'*60}")
            if metric == "mcc":
                trained_results = evaluate_sae(
                    sae_name, sae_model, c_embeddings, c_trained_acts, c_labels, args.linear_probe
                )
                trained_dec_results = compute_decoder_projection_mcc(
                    sae_model, c_embeddings, c_labels
                )
                print(f"  Decoder MCC:    {trained_dec_results['mcc']:.4f} (feature {trained_dec_results['feature_idx']})")

                # Collect for aggregation
                if args.aggregate_seeds:
                    model_type = _get_model_type(sae_name)
                    aggregated_results.setdefault(model_type, []).append(trained_results['mcc'])

                # Compare with first SSAE
                if first_ssae_results:
                    print(f"\n  vs SSAE:")
                    print(f"    MCC diff:     {first_ssae_results['mcc'] - trained_results['mcc']:.4f} (SSAE - {sae_name})")
                    print(f"    Steer diff:   {first_ssae_results['steer']['mean_cos_sim'] - trained_results['steer']['mean_cos_sim']:.4f}")
                    if first_ssae_dec_results:
                        print(f"    Dec MCC diff: {first_ssae_dec_results['mcc'] - trained_dec_results['mcc']:.4f} (SSAE - {sae_name})")
            else:
                trained_results = evaluate_sae_dci(sae_name, c_trained_acts, c_labels)
                if first_ssae_results:
                    print(f"\n  vs SSAE:")
                    print(f"    Disent diff:  {first_ssae_results['disentanglement'] - trained_results['disentanglement']:.4f} (SSAE - {sae_name})")
                    print(f"    Compl diff:   {first_ssae_results['completeness'] - trained_results['completeness']:.4f}")

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
                if first_ssae_results:
                    print(f"\n  vs SSAE:")
                    print(f"    MCC diff:     {first_ssae_results['mcc'] - pretrained_results['mcc']:.4f} (SSAE - Pretrained)")
                    print(f"    Steer diff:   {first_ssae_results['steer']['mean_cos_sim'] - pretrained_results['steer']['mean_cos_sim']:.4f}")
                    if first_ssae_dec_results:
                        print(f"    Dec MCC diff: {first_ssae_dec_results['mcc'] - pretrained_dec_results['mcc']:.4f} (SSAE - Pretrained)")
            else:
                pretrained_results = evaluate_sae_dci("Pretrained SAE", c_pretrained_acts, c_labels)
                if first_ssae_results:
                    print(f"\n  vs SSAE:")
                    print(f"    Disent diff:  {first_ssae_results['disentanglement'] - pretrained_results['disentanglement']:.4f} (SSAE - Pretrained)")
                    print(f"    Compl diff:   {first_ssae_results['completeness'] - pretrained_results['completeness']:.4f}")

        # --- Raw embeddings linear probe ---
        if args.linear_probe and metric == "mcc":
            print(f"\n{'-'*60}")
            print(f"Raw Embeddings Linear Probe — {concept_name}")
            print(f"{'-'*60}")
            probe_emb = compute_linear_probe_mcc(c_embeddings, c_labels)
            print(f"  Pearson corr: {probe_emb['mcc']:.4f}")
            print(f"  Accuracy:     {probe_emb['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Print aggregated summary if --aggregate-seeds
    # -------------------------------------------------------------------------
    if args.aggregate_seeds and aggregated_results and metric == "mcc":
        print(f"\n{'='*60}")
        print("AGGREGATED RESULTS (mean ± std over seeds)")
        print(f"{'='*60}")
        for model_type in sorted(aggregated_results.keys()):
            values = aggregated_results[model_type]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0.0
                print(f"  {model_type:12s}: MCC = {mean_val:.4f} ± {std_val:.4f}  (n={len(values)})")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
