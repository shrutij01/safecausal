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

# import debug_tools as dbg  # Commented out to fix sys.excepthook error


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
        self.input_norm = nn.LayerNorm(rep_dim)
        self.enc_norm = dict(
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.input_norm(x)
        # trainable layernorm at the input
        x_biased = x_norm - self.decoder.bias
        h = self.enc_norm(self.encoder(x_biased))

        # Decode and denormalize efficiently
        x_hat = self.decoder(h)
        return x_hat, h

    # remove denormalisation at the output


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

    @staticmethod
    def prepare_config(cfg, dataloader=None, device=None, model=None):
        """Prepare enhanced config with runtime info for wandb tracking."""
        config = asdict(cfg)

        # Convert Path objects to strings
        for k, v in config.items():
            if isinstance(v, Path):
                config[k] = str(v)

        # Add runtime info if provided
        if device:
            config["device"] = str(device)
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(device)
                config.update(
                    {
                        "gpu_name": gpu_props.name,
                        "gpu_memory_gb": gpu_props.total_memory / 1024**3,
                    }
                )

        if dataloader:
            config.update(
                {
                    "dataset_size": len(dataloader.dataset),
                    "num_batches": len(dataloader),
                }
            )

        # Add derived parameters
        config["hidden_dim"] = cfg.hid
        if hasattr(cfg, "extra") and hasattr(cfg.extra, "__dict__"):
            for k, v in vars(cfg.extra).items():
                config[f"extra_{k}"] = v

        return config

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


class SimpleCPUData(Dataset):
    def __init__(
        self,
        h5_path: str,
        key: str = "cfc_train",
        dataset_name: str = None,
        max_samples: int = None,
        data_seed: int = 21,
    ):
        with h5py.File(h5_path, "r") as f:
            full_data = f[key][:]  # Load entire dataset into RAM

        # Sample subset for labeled-sentences to speed up training
        if (
            dataset_name == "labeled-sentences"
            and max_samples is not None
            and len(full_data) > max_samples
        ):
            import numpy as np

            rng = np.random.RandomState(
                data_seed
            )  # Fixed data seed independent of model seed
            indices = rng.choice(
                len(full_data), size=max_samples, replace=False
            )
            self.data = full_data[indices]
            print(
                f"Sampled {max_samples} from {len(full_data)} samples using data_seed={data_seed}"
            )
        else:
            self.data = full_data

    def __len__(self):
        return (
            len(self.data) - 1 if len(self.data.shape) == 2 else len(self.data)
        )

    def __getitem__(self, idx):
        if len(self.data.shape) == 3:  # (N, 2, rep_dim)
            return torch.from_numpy(self.data[idx, 1] - self.data[idx, 0])
        else:  # (N, rep_dim)
            return torch.from_numpy(self.data[idx + 1] - self.data[idx])


KEYS = ("schedule", "oc", "seed", "target")  # choose what matters


def _hash_cfg(cfg) -> str:
    cfg_dict = asdict(cfg)
    # Convert Path objects and SimpleNamespace to JSON serializable types
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)  # Convert SimpleNamespace to dict
    blob = json.dumps(cfg_dict, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]  # short & stable


def dump_run(root: Path, model: torch.nn.Module, cfg) -> Path:
    from datetime import datetime

    # Extract dataset name from embedding file path
    dataset_name = cfg.emb.stem.split("_")[0]  # Get first part before '_'

    tag = "_".join(f"{k}{getattr(cfg, k)}" for k in KEYS)

    run = root / f"{dataset_name}_{tag}_{_hash_cfg(cfg)}"
    run.mkdir(
        parents=True, exist_ok=True
    )  # Allow multiple runs with same config

    torch.save(model.state_dict(), run / f"weights.pth")

    # Convert config to YAML-serializable format
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)

    (run / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
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
    concept_counts_gpu = torch.tensor(0, device=dev, dtype=torch.long)

    # Pre-allocated tensors to eliminate dynamic allocation
    gpu_tensor = None
    threshold_mask = None
    renorm_counter = 0

    # Get dimensions from dataset for pre-allocation
    batch_size = cfg.batch
    input_dim = dataloader.dataset.data.shape[-1]  # Get from dataset directly
    hid_dim = cfg.hid

    # Pre-allocate all working tensors on GPU (required)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for training")

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
            h_abs = torch.abs(h)

            # Use pre-allocated mask for sparsity counting
            torch.ge(h_abs, cfg.ind_th, out=threshold_mask)
            concept_counts_gpu += threshold_mask.sum()

    # Final cleanup and transfer concept counts to CPU once
    concept_counts = concept_counts_gpu.item()
    del gpu_tensor, threshold_mask, concept_counts_gpu
    torch.cuda.empty_cache()

    return recon_sum, sparsity_sum, concept_counts


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dict = make_dict(cfg).to(dev)

    # Prepare enhanced config with runtime info
    enhanced_config = Logger.prepare_config(cfg, dataloader, dev, dict)
    logger = Logger(
        entity="causalrepl", project="ssae", config=enhanced_config
    )

    ssae = make_ssae(cfg, dev)
    optim = make_optim(dict=dict, ssae=ssae, cfg=cfg)

    for ep in range(cfg.epochs):
        ssae.epoch = ep  # Update epoch for sparsity scheduling
        (
            total_recon_loss,
            total_sparsity_defect,
            total_active_concepts,
        ) = train_epoch(dataloader, dict, ssae, optim, cfg, dev)

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

        # Memory cleanup every 100 epochs for better performance
        if ep % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        return self.oc  # changed to rm dependence on n_concepts


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("emb", type=Path)
    add("data_cfg", type=Path)
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=15_000)
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
    # Extract dataset name from embedding file path
    dataset_name = cfg.emb.stem.split("_")[0] if "_" in cfg.emb.stem else None

    # Set max_samples for labeled-sentences to speed up training
    max_samples = 5500 if dataset_name == "labeled-sentences" else None

    dataset = SimpleCPUData(
        cfg.emb,
        key="cfc_train",
        dataset_name=dataset_name,
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=8,  # Match CPU cores for better utilization
        prefetch_factor=4,  # Higher prefetch for larger batches
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=True,  # Pin memory for faster GPU transfer
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
