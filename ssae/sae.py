"""
Standard Sparse Autoencoder (SAE) baseline for comparison with SSAE.

Trains on individual embeddings (z, tilde_z) rather than paired differences
(delta_z = tilde_z - z).  Uses the SAE's built-in sparsity mechanisms
(ReLU, TopK, JumpReLU, Sparsemax, MP) with type-specific regularisation,
cosine-warmup LR schedule, AdamW, and gradient clipping.
"""

import argparse
import math
import random
from typing import Any, Dict, Tuple
from types import SimpleNamespace
import hashlib, json, yaml
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import wandb

try:
    from sparsemax import Sparsemax
except ImportError:
    Sparsemax = None

from ssae.ssae import (
    DataWhitener,
    set_seeds,
    Logger,
    _load_yaml,
    renorm_decoder_cols_,
    project_decoder_grads_,
)


# ============================================================================
# UTILS
# ============================================================================


def rectangle(x):
    return ((x >= -0.5) & (x <= 0.5)).float()


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold, bandwidth)
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return x_grad, threshold_grad, None


def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)


def softplus_inverse(input, beta=1.0, threshold=20.0):
    """Inverse of the softplus function."""
    if isinstance(input, float):
        input = torch.tensor([input])
    if input * beta < threshold:
        return (1 / beta) * torch.log(torch.exp(beta * input) - 1.0)
    else:
        return input[0]


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(
                threshold, dtype=input.dtype, device=input.device
            )
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(
                bandwidth, dtype=input.dtype, device=input.device
            )
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0 * grad_output
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None


def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)


def update_cosine_warmup_lr(step, cfg, optimizer, total_steps):
    """Cosine schedule with linear warmup. Returns (step+1, current_lr)."""
    if step < cfg.warmup_iters:
        lr = cfg.lr * step / max(1, cfg.warmup_iters)
    else:
        progress = (step - cfg.warmup_iters) / max(
            1, total_steps - cfg.warmup_iters
        )
        lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return step + 1, lr


# ============================================================================
# SAE MODEL
# ============================================================================


class SAE(torch.nn.Module):
    def __init__(
        self,
        dimin=2,
        width=5,
        sae_type="relu",
        kval_topk=None,
        mp_kval=None,
        lambda_init=None,
    ):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'jumprelu', 'sparsemax_dist', 'MP'
        kval_topk: (int)
            k in topk sae_type
        mp_kval: (int)
            k in matching pursuit sae_type
        lambda_init: (float)
            initial value for lambda (only used by sparsemax/MP)
        """
        super(SAE, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.eps = 0.01

        # Lambda parameter (only meaningful for sparsemax/MP)
        lambda_init_val = 1.0 if lambda_init is None else lambda_init
        lambda_pre = softplus_inverse(lambda_init_val)
        self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False)

        # Encoder parameters — Kaiming uniform init
        self.Ae = nn.Parameter(torch.empty((width, dimin)))
        init.kaiming_uniform_(self.Ae, a=math.sqrt(5))
        if "MP" not in sae_type:
            self.be = nn.Parameter(torch.zeros((1, width)))

        # Decoder parameters — init from encoder, then normalize columns
        self.bd = nn.Parameter(torch.zeros((1, dimin)))
        if "MP" not in sae_type:
            self.Ad = nn.Parameter(self.Ae.data.T.clone())
            with torch.no_grad():
                renorm_decoder_cols_(self.Ad)

        # Parameters for specific SAEs
        # JumpReLU
        if sae_type == "jumprelu":
            self.logthreshold = nn.Parameter(
                torch.log(1e-3 * torch.ones((1, width)))
            )
            self.bandwidth = 1e-3

        # Sparsemax
        if "sparsemax" in sae_type:
            if Sparsemax is None:
                raise ImportError(
                    "sparsemax package required for sparsemax SAE types. "
                    "Install with: pip install sparsemax"
                )
            lambda_init_sm = (
                1 / (width * dimin) if lambda_init is None else lambda_init
            )
            lambda_pre_sm = softplus_inverse(lambda_init_sm)
            self.lambda_pre = nn.Parameter(lambda_pre_sm)

            with torch.no_grad():
                Ae_unit = self.Ae / (
                    self.eps + torch.linalg.norm(self.Ae, dim=1, keepdim=True)
                )
                self.Ae.copy_(Ae_unit)
                Ad_unit = (
                    self.Ad
                    / (
                        self.eps
                        + torch.linalg.norm(self.Ad, dim=1, keepdim=True)
                    )
                    * 48.0
                )
                self.Ad.copy_(Ad_unit)

        # Topk parameter
        if sae_type == "topk":
            self.kval_topk = kval_topk

        # MP parameter
        if sae_type == "MP":
            self.mp_kval = mp_kval

    @property
    def lambda_val(self):
        return F.softplus(self.lambda_pre)

    def forward(self, x, return_hidden=False, inf_k=None):
        # Vanilla ReLU — standard SAE (Bricken et al.)
        if self.sae_type == "relu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            codes = F.relu(x)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # TopK — structural sparsity (Gao et al.)
        elif self.sae_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            topk_values, topk_indices = torch.topk(x, kval, dim=-1)
            codes = torch.zeros_like(x)
            codes.scatter_(-1, topk_indices, F.relu(topk_values))
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # Matching Pursuits
        elif self.sae_type == "MP":
            lam = self.lambda_val
            kval = self.mp_kval if inf_k is None else inf_k
            x = x - self.bd
            codes = torch.zeros(x.shape[0], self.Ae.shape[0], device=x.device)
            for _ in range(kval):
                z = x @ self.Ae.T
                val, idx = torch.max(z, dim=1)
                to_add = torch.nn.functional.one_hot(
                    idx, num_classes=codes.shape[1]
                ).float()
                to_add = to_add * val.unsqueeze(1)
                to_add = to_add * lam
                codes = codes + to_add
                x = x - to_add @ self.Ae
            x = torch.matmul(codes, self.Ae) + self.bd

        # JumpReLU — learnable threshold (Rajamanoharan et al.)
        elif self.sae_type == "jumprelu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(x)
            threshold = torch.exp(self.logthreshold)
            codes = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # Sparsemax distance-based
        elif self.sae_type == "sparsemax_dist":
            lam = self.lambda_val
            x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            ae_norm_sq = torch.sum(self.Ae**2, dim=-1, keepdim=True).T
            dot_product = torch.matmul(x, self.Ae.T)
            x = -lam * (x_norm_sq + ae_norm_sq - 2 * dot_product)
            sm = Sparsemax(dim=-1)
            codes = sm(x)
            x = torch.matmul(codes, self.Ad.T)

        else:
            raise ValueError(f"Invalid sae_type: {self.sae_type!r}")

        if not return_hidden:
            return x
        else:
            return x, codes

    def decode(self, codes):
        if self.sae_type == "relu":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "topk":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "MP":
            x = torch.matmul(codes, self.Ae) + self.bd
        elif self.sae_type == "jumprelu":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "sparsemax_dist":
            x = torch.matmul(codes, self.Ad.T)
        return x

    def _reset_parameters(self):
        init.zeros_(self.Ae)
        init.zeros_(self.Ad)
        init.zeros_(self.be)
        init.zeros_(self.bd)


# ============================================================================
# DATASET  --  returns individual embeddings, not paired differences
# ============================================================================


class SimpleCPUDataSAE(Dataset):
    """
    Returns individual embeddings instead of paired differences.

    For (N, 2, rep_dim) data: returns both z and tilde_z as separate samples
        (dataset size = 2N).
    For (N, rep_dim) data: returns each embedding individually
        (dataset size = N).
    """

    def __init__(
        self,
        h5_path: str,
        key: str = "cfc_train",
        dataset_name: str = None,
        max_samples: int = None,
        data_seed: int = 21,
    ):
        with h5py.File(h5_path, "r") as f:
            full_data = f[key][:]

        # Sample subset for behavioral datasets to speed up training
        if (
            dataset_name in ["labeled-sentences", "sycophancy"]
            and max_samples is not None
            and len(full_data) > max_samples
        ):
            rng = np.random.RandomState(data_seed)
            orig_size = len(full_data)
            indices = rng.choice(orig_size, size=max_samples, replace=False)
            full_data = full_data[indices]
            print(
                f"Sampled {max_samples} from {orig_size} samples "
                f"for {dataset_name} using data_seed={data_seed}"
            )

        # Flatten paired data into individual embeddings
        if len(full_data.shape) == 3:  # (N, 2, rep_dim)
            self.data = np.concatenate(
                [full_data[:, 0], full_data[:, 1]], axis=0
            )
        else:  # (N, rep_dim)
            self.data = full_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ============================================================================
# CONFIG
# ============================================================================

KEYS = ["seed", "model", "oc", "sae_type"]


def _hash_cfg(cfg) -> str:
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)

    filtered_dict = {k: cfg_dict.get(k) for k in KEYS if k in cfg_dict}
    if hasattr(cfg, "extra") and hasattr(cfg.extra, "model"):
        filtered_dict["model"] = cfg.extra.model

    blob = json.dumps(filtered_dict, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]


@dataclass(frozen=True)
class Cfg:
    emb: Path
    batch: int = 32
    epochs: int = 20_000
    lr: float = 5e-4
    oc: int = -1  # -1 → use rep_dim from YAML config (embedding_dim)
    sae_type: str = "relu"
    kval_topk: int = 10
    mp_kval: int = 5
    gamma_reg: float = -1.0  # -1 → use default (1e-5 jumprelu, 1e-4 otherwise)
    lambda_init: float = None
    loss: str = "absolute"
    ind_th: float = 0.1
    seed: int = 0
    use_amp: bool = True
    quick: bool = False
    encoder_reg: bool = True  # sparsemax: use encoder weights in dist penalty

    renorm_epochs: int = 50  # renormalize decoder columns every N epochs (0 = off)

    # AdamW parameters
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 0.0  # 0 = no clipping
    warmup_iters: int = 0  # linear warmup steps before cosine decay

    # Whitening parameters
    whiten: bool = False
    whiten_epsilon: float = 1e-5

    # spill-over lives here (read-only)
    extra: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def hid(self) -> int:
        return self.oc


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("emb", type=Path)
    add("data_cfg", type=Path)
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=20_000)
    add("--lr", type=float, default=5e-4)
    add("--oc", type=int, default=-1,
        help="Hidden dim (-1 = use embedding_dim, capped at embedding_dim)")
    add(
        "--sae-type",
        default="relu",
        choices=["relu", "topk", "jumprelu", "sparsemax_dist", "MP"],
    )
    add("--kval-topk", type=int, default=10)
    add("--mp-kval", type=int, default=5)
    add(
        "--gamma-reg",
        type=float,
        default=-1.0,
        help="Regularisation weight (-1 = auto: 1e-5 jumprelu, 1e-4 otherwise)",
    )
    add("--lambda-init", type=float, default=None)
    add("--loss", default="absolute", choices=["relative", "absolute"])
    add("--ind-th", type=float, default=0.1)
    add("--seed", type=int, default=0)
    add("--use-amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use mixed precision (AMP). Disable with --no-use-amp")
    add(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller dataset for quick training",
    )
    add(
        "--encoder-reg",
        action="store_true",
        default=True,
        help="Sparsemax: use encoder weights in distance penalty",
    )
    add(
        "--no-encoder-reg",
        dest="encoder_reg",
        action="store_false",
        help="Sparsemax: use decoder weights in distance penalty",
    )
    add("--renorm-epochs", type=int, default=50,
        help="Renormalize decoder columns every N epochs (0=off)")
    # AdamW
    add("--weight-decay", type=float, default=0.0)
    add("--beta1", type=float, default=0.9)
    add("--beta2", type=float, default=0.999)
    add("--grad-clip", type=float, default=0.0, help="Max grad norm (0=off)")
    add("--warmup-iters", type=int, default=0, help="LR linear warmup steps")
    # Whitening
    add(
        "--whiten",
        action="store_true",
        help="Apply ZCA whitening to input embeddings",
    )
    add("--whiten-epsilon", type=float, default=1e-5)
    cli: Dict[str, Any] = vars(p.parse_args())

    yaml_path = cli.pop("data_cfg")
    yaml_cfg = _load_yaml(yaml_path)
    field_names = {f.name for f in fields(Cfg)}
    extra = {k: v for k, v in yaml_cfg.items() if k not in field_names}
    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", SimpleNamespace(**extra))

    # Resolve oc: default to rep_dim (embedding_dim), cap at rep_dim
    rep_dim = cfg.extra.rep_dim
    if cfg.oc <= 0:
        object.__setattr__(cfg, "oc", rep_dim)
    elif cfg.oc > rep_dim:
        print(f"Warning: oc={cfg.oc} > rep_dim={rep_dim}, capping to {rep_dim}")
        object.__setattr__(cfg, "oc", rep_dim)

    return cfg


# ============================================================================
# DATA LOADING
# ============================================================================


def make_dataloader(cfg: Cfg) -> DataLoader:
    dataset_name = cfg.emb.stem.split("_")[0] if "_" in cfg.emb.stem else None

    allowed_quick_datasets = [
        "labeled-sentences",
        "sycophancy",
        "bias-in-bios",
        "labeled-sentences-correlated",
    ]
    use_quick = cfg.quick and dataset_name in allowed_quick_datasets
    if cfg.quick and not use_quick:
        print(
            f"Warning: --quick mode not supported for '{dataset_name}', "
            f"using full dataset"
        )

    max_samples = 5500 if use_quick else None

    dataset = SimpleCPUDataSAE(
        cfg.emb,
        key="cfc_train",
        dataset_name=dataset_name,
        max_samples=max_samples,
    )

    if cfg.whiten:
        print(
            f"Applying ZCA whitening to embeddings "
            f"(shape: {dataset.data.shape})..."
        )
        whitener = DataWhitener(epsilon=cfg.whiten_epsilon)
        dataset.data = whitener.fit_transform(dataset.data)

        save_dir = cfg.emb.parent / "run_out"
        save_dir.mkdir(parents=True, exist_ok=True)
        whitening_path = save_dir / "whitening_params.npz"
        whitener.save(whitening_path)
        print(f"Saved whitening parameters to {whitening_path}")

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {cfg.batch}")
    print(f"Number of batches: {len(dataset) // cfg.batch}")

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    print(f"DataLoader length: {len(train_loader)}")
    train_loader._dataset_ref = dataset
    return train_loader


# ============================================================================
# MODEL FACTORY
# ============================================================================


def make_model(cfg: Cfg) -> SAE:
    return SAE(
        dimin=cfg.extra.rep_dim,
        width=cfg.hid,
        sae_type=cfg.sae_type,
        kval_topk=cfg.kval_topk,
        mp_kval=cfg.mp_kval,
        lambda_init=cfg.lambda_init,
    )


# ============================================================================
# TYPE-SPECIFIC REGULARISATION
# ============================================================================


def compute_reg_loss(
    model: SAE,
    x: Tensor,
    codes: Tensor,
    encoder_reg: bool,
    dev: torch.device,
) -> Tensor:
    """Return the regularisation loss appropriate for each SAE type."""

    # ReLU / MP: L1 norm per sample, averaged over batch
    if model.sae_type in ("relu", "MP"):
        return torch.norm(codes, p=1, dim=-1).mean()

    # Sparsemax: distance-based penalty
    if model.sae_type == "sparsemax_dist":
        if encoder_reg:
            dist = (x.unsqueeze(1) - model.Ae.unsqueeze(0)).pow(2).sum(dim=-1)
        else:
            dist = (
                (x.unsqueeze(1) - model.Ad.T.unsqueeze(0)).pow(2).sum(dim=-1)
            )
        return (dist * codes).sum(dim=-1).mean()

    # JumpReLU: L0 via differentiable step function
    if model.sae_type == "jumprelu":
        bandwidth = 1e-3
        return torch.mean(
            torch.sum(
                step_fn(codes, torch.exp(model.logthreshold), bandwidth),
                dim=-1,
            )
        )

    # TopK: sparsity is structural, no regularisation needed
    if model.sae_type == "topk":
        return torch.tensor(0.0, device=dev)

    raise ValueError(f"Unknown sae_type: {model.sae_type!r}")


# ============================================================================
# TRAINING
# ============================================================================

_LOSS: dict[str, callable] = {
    "relative": lambda x, x_hat: (
        torch.sum((x_hat - x).pow(2), dim=1)
        / (torch.sum(x.pow(2), dim=1) + 1e-8)
    ).mean(),
    "absolute": lambda x, x_hat: F.mse_loss(x_hat, x, reduction="mean"),
}


def train_epoch(
    dataloader: DataLoader,
    model: SAE,
    optimizer: torch.optim.Optimizer,
    cfg: Cfg,
    dev: torch.device,
    gamma_reg: float,
    global_step: int,
    total_steps: int,
) -> Tuple[float, float, float, int, float]:
    """Returns (recon_sum, reg_sum, concept_counts, global_step, last_lr)."""
    model.train()

    recon_sum = 0.0
    reg_sum = 0.0
    concept_counts_gpu = torch.tensor(0, device=dev, dtype=torch.long)
    lr = 0.0

    batch_size = cfg.batch
    input_dim = dataloader.dataset.data.shape[-1]
    hid_dim = cfg.hid

    gpu_tensor = torch.empty(
        (batch_size, input_dim), device=dev, dtype=torch.float32
    )
    threshold_mask = torch.empty(
        (batch_size, hid_dim), device=dev, dtype=torch.bool
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if cfg.use_amp and torch.cuda.is_available()
        else None
    )

    for batch_idx, x_cpu in enumerate(dataloader):
        # Cosine warmup LR update (per step)
        global_step, lr = update_cosine_warmup_lr(
            global_step, cfg, optimizer, total_steps
        )

        gpu_tensor.copy_(x_cpu, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                x_hat, codes = model(gpu_tensor, return_hidden=True)
                loss_mse = _LOSS[cfg.loss](gpu_tensor, x_hat)
                loss_reg = compute_reg_loss(
                    model, gpu_tensor, codes, cfg.encoder_reg, dev
                )
                loss = loss_mse + gamma_reg * loss_reg
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            # Project decoder gradients before step (preserve unit-norm columns)
            if hasattr(model, "Ad") and model.Ad.grad is not None:
                with torch.no_grad():
                    project_decoder_grads_(model.Ad)
            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat, codes = model(gpu_tensor, return_hidden=True)
            loss_mse = _LOSS[cfg.loss](gpu_tensor, x_hat)
            loss_reg = compute_reg_loss(
                model, gpu_tensor, codes, cfg.encoder_reg, dev
            )
            loss = loss_mse + gamma_reg * loss_reg
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            # Project decoder gradients before step (preserve unit-norm columns)
            if hasattr(model, "Ad") and model.Ad.grad is not None:
                with torch.no_grad():
                    project_decoder_grads_(model.Ad)
            optimizer.step()

        recon_sum += loss_mse.item()
        reg_sum += loss_reg.item()

        with torch.no_grad():
            h_abs = torch.abs(codes)
            torch.ge(h_abs, cfg.ind_th, out=threshold_mask)
            concept_counts_gpu += threshold_mask.sum()

    concept_counts = concept_counts_gpu.item()
    del gpu_tensor, threshold_mask, concept_counts_gpu
    torch.cuda.empty_cache()

    return recon_sum, reg_sum, concept_counts, global_step, lr


def dump_run(root: Path, model: torch.nn.Module, cfg: Cfg, gamma_reg: float) -> Path:
    from ssae.ssae import _extract_dataset_name, _extract_model_name

    dataset_name = _extract_dataset_name(cfg.emb.stem)
    model_name = _extract_model_name(cfg)

    # SAE folder: dataset, LLM type, sae type, oc, k-val, gamma-reg, seed
    # Format gamma_reg: e.g. 0.01 -> "0.01", resolved from -1 if needed
    gamma_str = f"{gamma_reg:g}"
    kval_str = f"{cfg.kval_topk}" if cfg.sae_type == "topk" else f"{cfg.mp_kval}" if cfg.sae_type == "MP" else "0"

    run = root / (
        f"sae_{dataset_name}_{model_name}"
        f"_{cfg.sae_type}"
        f"_oc{cfg.oc}"
        f"_k{kval_str}"
        f"_g{gamma_str}"
        f"_seed{cfg.seed}"
    )
    run.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run / "weights.pth")

    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)

    (run / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    return run


# ============================================================================
# MAIN
# ============================================================================


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = make_model(cfg).to(dev)

    enhanced_config = Logger.prepare_config(cfg, dataloader, dev, model)
    logger = Logger(entity="causalrepl", project="sae", config=enhanced_config)

    # AdamW with configurable betas and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Resolve gamma_reg: -1 means use type-dependent default
    gamma_reg = cfg.gamma_reg
    if gamma_reg < 0:
        gamma_reg = 1e-5 if cfg.sae_type == "jumprelu" else 1e-4
    print(f"gamma_reg = {gamma_reg}  (sae_type={cfg.sae_type})")

    # Total optimizer steps across all epochs (for cosine schedule)
    steps_per_epoch = len(dataloader)
    total_steps = cfg.epochs * steps_per_epoch
    global_step = 0

    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.warmup_iters}")

    for ep in range(cfg.epochs):
        (
            total_recon_loss,
            total_reg_penalty,
            total_active_concepts,
            global_step,
            lr,
        ) = train_epoch(
            dataloader,
            model,
            optimizer,
            cfg,
            dev,
            gamma_reg,
            global_step,
            total_steps,
        )

        # Periodic decoder column renormalization (matches SSAE convention)
        if (
            cfg.renorm_epochs > 0
            and ep % cfg.renorm_epochs == 0
            and hasattr(model, "Ad")
        ):
            with torch.no_grad():
                renorm_decoder_cols_(model.Ad)

        num_batches = len(dataloader)
        dataset_size: int = len(dataloader.dataset)

        p_sparsity = total_active_concepts / (dataset_size * cfg.hid)

        epoch_metrics = {
            "epoch": ep,
            "recon_loss": total_recon_loss / num_batches,
            "reg_loss": total_reg_penalty / num_batches,
            "l0_sparsity": total_active_concepts / dataset_size,
            "p_sparsity": p_sparsity,
            "lambda": model.lambda_val.item(),
            "lr": lr,
        }

        for k, v in epoch_metrics.items():
            logger.logkv(k, v)

        if ep % 10 == 0:
            logger.log_memory_stats()

        if ep % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.dumpkvs()
        print(
            f"ep {ep:04d}  recon {epoch_metrics['recon_loss']:.4f}  "
            f"reg {epoch_metrics['reg_loss']:.4f}  "
            f"l0 {epoch_metrics['l0_sparsity']:.4f}  "
            f"lr {lr:.6f}"
        )

    dump_run(cfg.emb.parent / "run_out", model, cfg, gamma_reg)


if __name__ == "__main__":
    main()
