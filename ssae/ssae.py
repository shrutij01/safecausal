import argparse
import random
from typing import Any, Dict, Tuple
from types import SimpleNamespace
from matplotlib.pyplot import box
from traitlets import observe
import hashlib, json, yaml
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields


import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import wandb
import cooper
from box import Box

import debug_tools as dbg


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """
    Initialise every RNG we care about.

    Args
    ----
    seed : int
        Non-negative seed.
    deterministic : bool, default=True
        Toggle PyTorch deterministic algorithms and turn off CUDNN bench-
        mark mode (bench = speed, determinism = repro).

    Notes
    -----
    • `torch.cuda.manual_seed_all` is a no-op on CPU-only boxes.
    • We do *not* touch env-vars here (caller decides).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # reduces speed so only use when need bit-perfect reproducibility
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def layer_normalise_(
    x: Tensor, *, eps: float = 1e-5
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Memory-efficient layer normalization over the last dimension.
    """
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = (var + eps).rsqrt_()  # in-place rsqrt for efficiency

    # Fused normalize operation
    x_norm = x.sub(mu).mul_(inv_std)  # sub creates new tensor, mul_ in-place

    return x_norm, {"mu": mu, "inv_std": inv_std}


@torch.jit.script
def fused_layer_norm_inplace_(x: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Ultra-fast in-place layer normalization for training loops.
    """
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = (var + eps).rsqrt_()

    # In-place normalization
    x.sub_(mu).mul_(inv_std)
    return x


class DictLinearAE(nn.Module):
    """
    Implements:
        coefficients = encoder(delta_z) + encoder_bias
        recons = decoder(coefficients) + decoder_bias
        delta_z --> rep_dim x 1, coefficients --> num_concepts x 1
        encoder --> num_concepts x rep_dim, decoder --> rep_dim x num_concepts
        encoder_bias --> num_concepts x 1, decoder_bias --> rep_dim x 1
    """

    def __init__(
        self,
        rep_dim: int,
        hid: int,
        norm_type: str,
    ) -> None:
        super(DictLinearAE, self).__init__()
        self.encoder = nn.Linear(rep_dim, hid, bias=True)
        self.norm = dict(
            ln=nn.LayerNorm(hid),
            gn=nn.GroupNorm(1, hid),
            bn=nn.BatchNorm1d(hid),
        ).get(norm_type) or self._bad_norm(norm_type)

        self.decoder = nn.Linear(hid, rep_dim, bias=True)
        # copy, do NOT tie; we only want identical init
        self.decoder.weight.data.copy_(self.encoder.weight.T)
        renorm_decoder_cols_(self.decoder.weight)

    @staticmethod
    def _bad_norm(nt):
        raise ValueError(f"norm_type {nt!r} not in ln/gn/bn")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # In-place bias subtraction to avoid copy
        x_enc = x.clone()  # Clone to avoid modifying input
        x_enc.sub_(self.decoder.bias)
        return self.norm(self.encoder(x_enc))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Efficient single-pass forward with minimal allocations
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = (var + 1e-5).rsqrt_()  # in-place rsqrt for efficiency

        # Create normalized version without modifying input
        x_norm = (x - mu) * inv_std

        # Fused encode: subtract decoder bias and encode in one operation
        x_biased = x_norm - self.decoder.bias
        h = self.norm(self.encoder(x_biased))

        # Decode and denormalize efficiently
        x_hat = self.decoder(h)
        x_hat.mul_(inv_std).add_(mu)  # In-place denormalization

        return x_hat, h


# ----------------------------------------------------------------------
# 1. Hard projection: W ← W / ||W||₂ column-wise
# ----------------------------------------------------------------------
@torch.jit.script
def renorm_decoder_cols_(W: Tensor, eps: float = 1e-8) -> None:
    """
    In-place column l_2 normalisation.
    Safe for zero columns (leaves them unchanged).
    """
    col_norms = W.norm(p=2, dim=0, keepdim=True).clamp_min_(eps)
    W.data.div_(col_norms)


# ----------------------------------------------------------------------
# 2. Project decoder gradients to the tangent space of unit-norm columns
#    g ← g − (wᵀg) w     (vectorised, no Python loop)
# ----------------------------------------------------------------------
@torch.jit.script
def project_decoder_grads_(W: Tensor) -> None:
    """
    In-place orthogonal-projection of ∇W so future SGD preserves
    unit-norm columns.  Assumes `W.grad` exists and `W` is already
    normalised.
    """
    g = W.grad
    if g is None:
        return  # nothing to do

    # (wᵀg) for each column  → shape (1, C)
    col_dot = (W * g).sum(dim=0, keepdim=True)
    # g ← g − w (wᵀg)
    g.sub_(W * col_dot)


class SSAE(cooper.ConstrainedMinimizationProblem):
    """MSE + l_1 sparsity constraint on a linear schedule."""

    _LOSS: dict[str, callable] = {
        "relative": lambda z, z_hat: (
            torch.sum((z_hat - z).pow(2), dim=1) / torch.sum(z.pow(2), dim=1)
        ).mean(),  # Vectorized relative loss computation
        "absolute": lambda z, z_hat: nn.functional.mse_loss(
            z_hat, z, reduction="mean"
        ),
    }

    def __init__(
        self,
        dev: torch.device,  # Device for the model
        *,
        batch: int,
        hid: int,
        n_concepts: int,
        warmup: int = 2_000,
        schedule: int = 5_000,
        target: float = 0.10,
    ) -> None:
        super().__init__()
        self.batch, self.hid = batch, hid

        # sparsity schedule
        self._lvl0 = n_concepts
        self._lvl1 = target
        self._warm = warmup
        self._T = max(schedule, 1)
        self.level = n_concepts

        self.epoch = 0

        # Cache last forward pass results to avoid recomputation
        self._cached_h: torch.Tensor | None = None
        self.last_cmp_state: cooper.CMPState | None = None
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1).to(
            dev
        )

        self.sparsity_constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def _tick_schedule(self) -> None:
        if self.epoch < self._warm:
            return
        t = (self.epoch - self._warm) / self._T
        self.level = self._lvl0 + min(1.0, t) * (self._lvl1 - self._lvl0)

    def compute_cmp_state(
        self,
        model: nn.Module,
        delta_z: Tensor,
        loss_type: str = "relative",
    ) -> cooper.CMPState:
        """Return CMPState used by Cooper’s optimiser."""
        delta_z_hat, h = model(delta_z)
        loss = self._LOSS[loss_type](delta_z, delta_z_hat)

        # Cache hidden states for sparsity computation
        self._cached_h = h
        actual_batch_size = h.shape[0]

        self._tick_schedule()
        ineq = h.abs().sum() / (actual_batch_size * self.hid) - self.level
        constraint_state = cooper.ConstraintState(violation=ineq)
        observed_constraints = {self.sparsity_constraint: constraint_state}
        cmp_state = cooper.CMPState(
            loss=loss, observed_constraints=observed_constraints
        )
        self.last_cmp_state = cmp_state  # Cache for metrics
        return cmp_state

    # ──────────────────────────────────────────────────────────────
    def metrics(self) -> tuple[float, float]:
        """(loss, ineq) as floats – call *after* the last closure."""
        if self.last_cmp_state is None:
            raise RuntimeError(
                "metrics() called before compute_cmp_state(). No cached state available."
            )
        state = self.last_cmp_state
        return float(state.loss), float(
            state.observed_constraints[self.sparsity_constraint].violation
        )

    def get_cached_hidden(self) -> Tensor:
        """Get hidden states from last forward pass to avoid recomputation."""
        if self._cached_h is None:
            raise RuntimeError(
                "get_cached_hidden() called before compute_cmp_state()"
            )
        return self._cached_h


class Logger:
    "A thin, side-effect-free wrapper around wandb with memory monitoring."

    def __init__(self, **kws):
        self._run = wandb.init(
            settings=wandb.Settings(code_dir="."),
            dir="./wandb",
            **kws,
        )
        self._buf: Dict[str, float] = {}

    def logkv(self, k, v):
        self._buf[k] = float(v) if torch.is_tensor(v) else v
        return v

    def log_memory_stats(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            self._buf.update(
                {
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
                    / 1024**2,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved()
                    / 1024**2,
                    "gpu_memory_cached_mb": torch.cuda.memory_cached()
                    / 1024**2,
                }
            )

    def dumpkvs(self):
        if self._buf:
            self._run.log(self._buf)
            self._buf = {}


class LazyCPUData(Dataset):
    """
    Memory-optimized H5PY dataset with automatic resource management.
    Uses per-worker file handles and pre-allocated tensor buffers.
    """

    def __init__(self, h5_path: str, key: str = "cfc_train"):
        self.h5_path = str(h5_path)
        self.key = key
        self._file_handle = None
        self._len = None
        self._rep_dim = None

        # Get metadata without keeping file open
        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            if key not in f or not isinstance(f[key], h5py.Dataset):
                raise TypeError(f"{key} is not an HDF5 dataset")
            dataset_shape = f[key].shape
            if len(dataset_shape) == 3:
                self._rep_dim = dataset_shape[2]  # (N, T, rep_dim)
            elif len(dataset_shape) == 2:
                self._rep_dim = dataset_shape[1]  # (N, rep_dim)
            else:
                raise ValueError(
                    f"Unexpected shape {dataset_shape} for dataset {key}"
                )
            self._len = len(f[key])

        # Pre-allocate numpy buffer for zero-copy operations
        # expects two samples to take diff between
        self._buffer = np.empty((2, self._rep_dim), dtype=np.float32)
        self._delta_buffer = np.empty(self._rep_dim, dtype=np.float32)

    def _get_file_handle(self):
        """Lazy per-worker file handle creation."""
        if self._file_handle is None:
            self._file_handle = h5py.File(
                self.h5_path,
                "r",
                libver="latest",
                swmr=True,  # Single-writer multiple-reader mode
                rdcc_nbytes=1024 * 1024 * 16,  # 16MB cache per file
            )
        return self._file_handle

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        ds = self._get_file_handle()[self.key]

        # Read directly into pre-allocated buffer to avoid extra allocation
        ds.read_direct(self._buffer, np.s_[idx])

        # In-place difference using pre-allocated buffer
        np.subtract(self._buffer[1], self._buffer[0], out=self._delta_buffer)

        # Return tensor that shares memory with numpy buffer
        return torch.from_numpy(
            self._delta_buffer.copy()
        )  # Copy to avoid data races

    def __del__(self):
        if self._file_handle is not None:
            self._file_handle.close()


KEYS = ("schedule", "oc", "seed")  # choose what matters


def _hash_cfg(cfg) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]  # short & stable


def dump_run(root: Path, model: torch.nn.Module, cfg) -> Path:
    tag = "_".join(f"{k}{getattr(cfg, k)}" for k in KEYS)
    run = root / f"{tag}_{_hash_cfg(cfg)}"
    run.mkdir(parents=True, exist_ok=False)

    torch.save(model.state_dict(), run / "weights.pth")
    (run / "cfg.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    return run


def train_epoch(
    dataloader,
    dict,
    ssae,
    optim,
    cfg,
    dev,
) -> Tuple[float, float, float]:
    dict.train()

    # Accumulators
    recon_sum = 0.0
    sparsity_sum = 0.0
    concept_counts = 0

    # Pre-allocated tensors to eliminate dynamic allocation
    gpu_tensor = None
    threshold_mask = None
    batch_count = 0
    renorm_counter = 0

    # Get batch and hidden dimensions for pre-allocation
    sample_batch = next(iter(dataloader))
    batch_size, input_dim = sample_batch.shape
    hid_dim = cfg.hid

    # Pre-allocate all working tensors on GPU
    if torch.cuda.is_available():
        gpu_tensor = torch.empty(
            (batch_size, input_dim), device=dev, dtype=torch.float32
        )
        threshold_mask = torch.empty(
            (batch_size, hid_dim), device=dev, dtype=torch.bool
        )

    # Mixed precision scaler for 2x speedup
    scaler = (
        torch.cuda.amp.GradScaler()
        if cfg.use_amp and torch.cuda.is_available()
        else None
    )

    for batch_idx, delta_z_cpu in enumerate(dataloader):
        with torch.cuda.device(dev):
            # Efficient GPU transfer with pre-allocated tensor
            gpu_tensor.copy_(delta_z_cpu, non_blocking=True)

            # Zero gradients before forward pass
            optim_kwargs = {
                "model": dict,
                "delta_z": gpu_tensor,
                "loss_type": cfg.loss,
            }

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    optim.roll(compute_cmp_state_kwargs=optim_kwargs)
                # Note: Cooper handles its own scaling internally
            else:
                optim.roll(compute_cmp_state_kwargs=optim_kwargs)

            # Post-step weight operations
            with torch.no_grad():
                if dict.decoder.weight.grad is not None:
                    project_decoder_grads_(dict.decoder.weight)

                # Renormalize decoder columns at configured frequency
                renorm_counter += 1
                if renorm_counter % cfg.renorm_epochs == 0:
                    renorm_decoder_cols_(dict.decoder.weight)

            # Extract metrics
            recon_loss, sparsity_penalty = ssae.metrics()
            recon_sum += recon_loss
            sparsity_sum += sparsity_penalty

            # Efficient sparsity counting using cached hidden states
            with torch.no_grad():
                h = ssae.get_cached_hidden()  # Reuse from forward pass

                # Use pre-allocated mask for sparsity counting
                torch.ge(torch.abs(h), cfg.ind_th, out=threshold_mask)
                concept_counts += threshold_mask.sum().item()

            batch_count += 1

            # Memory cleanup every 100 batches (less frequent, more efficient)
            if batch_count % 100 == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    del gpu_tensor, threshold_mask
    torch.cuda.empty_cache()

    return recon_sum, sparsity_sum, concept_counts


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    logger = Logger(entity="causalrepl", project="ssae", config=asdict(cfg))
    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dict = make_dict(cfg).to(dev)
    ssae = make_ssae(cfg, dev)
    optim = make_optim(dict=dict, ssae=ssae, cfg=cfg)

    for ep in range(cfg.epochs):
        ssae.epoch = ep  # Update epoch for sparsity scheduling
        total_recon_loss, total_sparsity_defect, total_active_concepts = (
            train_epoch(dataloader, dict, ssae, optim, cfg, dev)
        )

        # Calculate correct averages
        num_batches = len(dataloader)
        dataset_size: int = len(dataloader.dataset)

        # Log epoch metrics with correct normalization
        epoch_metrics = {
            "epoch": ep,
            "recon_loss": total_recon_loss
            / num_batches,  # Average loss per batch
            "l0_sparsity": total_active_concepts
            / dataset_size,  # Number of active neurons
            "sparsity_target": ssae.level,  # Current scheduled sparsity target
            "constraint_violation": total_sparsity_defect
            / num_batches,  # Same as sparsity_defect but clearer name
        }

        for k, v in epoch_metrics.items():
            logger.logkv(k, v)

        # Log memory stats every 10 epochs
        if ep % 10 == 0:
            logger.log_memory_stats()

        logger.dumpkvs()
        print(
            f"ep {ep:04d}  recon_loss {epoch_metrics['recon_loss']:.4f}  "
            f"sparsity_defect {epoch_metrics['constraint_violation']:.4f}  "
            f"target {epoch_metrics['sparsity_target']:.4f}  "
            f"l0 {epoch_metrics['l0_sparsity']:.4f}"
        )

    dump_run(cfg.emb.parent / "run_out", dict, cfg)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return Box(yaml.safe_load(fh) or {})


# ===== run config ============================================================
@dataclass(frozen=True)
class Cfg:
    emb: Path
    batch: int = 32
    epochs: int = 20_000
    lr: float = 5e-4
    oc: int = 10
    n_concepts: int = 1
    warmup: int = 2_000
    schedule: int = 5_000
    target: float = 0.1
    norm: str = "ln"
    loss: str = "relative"
    ind_th: float = 0.1
    seed: int = 0
    renorm_epochs: int = 50
    use_amp: bool = True

    # spill-over lives here (read-only)
    extra: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def hid(self) -> int:  # encoder dim
        return self.oc * self.n_concepts


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("emb", type=Path)
    add("data_cfg", type=Path)
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=9_000)
    add("--lr", type=float, default=5e-4)
    add("--oc", type=int, default=10)
    add("--n-concepts", "-C", type=int, default=1)
    add("--warmup", type=int, default=2_000)
    add("--schedule", type=int, default=1_000)
    add("--target", type=float, default=0.1)
    add("--norm", choices=["ln", "gn", "bn"], default="ln")
    add("--loss", default="relative", choices=["relative", "absolute"])
    add("--ind-th", type=float, default=0.1)
    add("--seed", type=int, default=0)
    add("--renorm-epochs", type=int, default=50)
    add("--use-amp", action="store_true", default=True)
    cli: Dict[str, Any] = vars(p.parse_args())

    yaml_path = cli.pop("data_cfg")
    yaml_cfg = _load_yaml(yaml_path)
    # ----- split YAML into (known, extra) -----------------------------------
    field_names = {f.name for f in fields(Cfg)}
    extra = {k: v for k, v in yaml_cfg.items() if k not in field_names}
    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", SimpleNamespace(**extra))

    return cfg


def make_dataloader(cfg) -> DataLoader:
    # Optimize for memory efficiency over throughput
    dataset = LazyCPUData(cfg.emb, key="cfc_train")

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=1,  # Reduced to minimize memory overhead
        prefetch_factor=1,  # Minimal prefetch to reduce memory
        persistent_workers=False,  # Let workers die to free memory
        pin_memory=torch.cuda.is_available(),  # Only if GPU available
        drop_last=True,  # Consistent batch sizes for memory pool
    )

    # Store dataset reference for cleanup
    train_loader._dataset_ref = dataset
    return train_loader


def make_dict(cfg: Cfg) -> torch.nn.Module:
    return DictLinearAE(cfg.extra.rep_dim, cfg.hid, cfg.norm)


def make_ssae(cfg: Cfg, dev: torch.device):
    return SSAE(
        dev=dev,
        batch=cfg.batch,
        hid=cfg.hid,
        n_concepts=cfg.n_concepts,
        warmup=cfg.warmup,
        schedule=cfg.schedule,
        target=cfg.target,
    )


def make_optim(dict: torch.nn.Module, ssae, cfg: Cfg):
    primal_optimizer = cooper.optim.ExtraAdam(dict.parameters(), lr=cfg.lr)
    dual_lr = cfg.lr / 2  # dual_lr should be less than primal_lr
    # 0.5 is common from extra gradient method literature
    # todo: how to select the dual optimizer?
    dual_optimizer = cooper.optim.ExtraAdam(
        ssae.dual_parameters(), lr=dual_lr, maximize=True
    )

    # Setup the constrained optimizer using Cooper's Lagrangian formulation
    coop_optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
        cmp=ssae,
        primal_optimizers=primal_optimizer,
        dual_optimizers=dual_optimizer,
    )
    return coop_optimizer


if __name__ == "__main__":
    # with dbg.debug_on_exception():
    main()
