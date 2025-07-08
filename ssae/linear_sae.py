import argparse
import random
from typing import Any, Dict, Tuple
from types import MappingProxyType
import hashlib, json, yaml
from pathlib import Path
from dataclasses import asdict, dataclass, field


import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import wandb
import cooper

import debug_tools as dbg


def set_seeds(seed: int, deterministic: bool = True) -> None:
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
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def layer_normalise_(
    x: Tensor, *, eps: float = 1e-5
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    In-place `(x - μ) / σ` over the *last* dim.

    Returns
    -------
    x    : same Tensor, normalised.
    stats: dict with 'mu' and 'inv_std'     (cheaper to reuse inv-σ).

    Complexity
    ----------
    O(N) memory-free; uses unbiased=False to avoid extra kernel.
    """
    mu = x.mean(dim=-1, keepdim=True)  # (⋯, 1)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = (var + eps).rsqrt_()  # numerically stable inverse-σ

    x.sub_(mu).mul_(inv_std)  # (x-μ)·σ⁻¹   (all in-place)
    return x, {"mu": mu, "inv_std": inv_std}


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

        self.decoder = nn.Linear(hid, rep_dim, bias=True, device=device)
        # copy, do NOT tie; we only want identical init
        self.decoder.weight.data.copy_(self.encoder.weight.T)
        self.to(device)
        renorm_decoder_cols_(self.decoder.weight)

    @staticmethod
    def _bad_norm(nt):
        raise ValueError(f"norm_type {nt!r} not in ln/gn/bn")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x -= self.decoder.bias
        return self.norm(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, stats = layer_normalise_(x)  # standardise
        h = self.encode(x)
        x_hat = self.decoder(h)
        x_hat.mul_(stats["inv_std"]).add_(stats["mu"])  # de-standardise
        return x_hat, h


# ----------------------------------------------------------------------
# 1. Hard projection: W ← W / ||W||₂ column-wise
# ----------------------------------------------------------------------
def renorm_decoder_cols_(W: Tensor, eps: float = 1e-8) -> None:
    """
    In-place column ℓ₂ normalisation.
    Safe for zero columns (leaves them unchanged).
    """
    col_norms = W.norm(dim=0, keepdim=True).clamp_min_(eps)
    W.div_(col_norms)


# ----------------------------------------------------------------------
# 2. Project decoder gradients to the tangent space of unit-norm columns
#    g ← g − (wᵀg) w     (vectorised, no Python loop)
# ----------------------------------------------------------------------
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
            nn.functional.mse_loss(z_hat, z, reduction="none").mean(1)
            / z.pow(2).mean(1)
        ).mean(),
        "absolute": lambda z, z_hat: nn.functional.mse_loss(
            z_hat, z, reduction="mean"
        ),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        batch: int,
        hid: int,
        n_concepts: int,
        warmup: int = 2_000,
        schedule: int = 5_000,
        target: float = 0.10,
    ) -> None:
        super().__init__(is_constrained=True)
        self.model = model
        self.batch, self.hid = batch, hid

        # sparsity schedule
        self._lvl0 = n_concepts
        self._lvl1 = target
        self._warm = warmup
        self._T = max(schedule, 1)
        self.level = n_concepts

        self.epoch = 0

    def _tick_schedule(self) -> None:
        if self.epoch < self._warm:
            return
        t = (self.epoch - self._warm) / self._T
        self.level = self._lvl0 + min(1.0, t) * (self._lvl1 - self._lvl0)

    def closure(
        self,
        z: Tensor,
        stats: dict[str, Tensor],
        *,
        loss_type: str = "relative",
    ) -> cooper.CMPState:
        """Return CMPState used by Cooper’s optimiser."""
        z_hat, h = self.model(z, stats)
        loss = self._LOSS[loss_type](z, z_hat)

        self._tick_schedule()
        ineq = h.abs().sum() / (self.batch * self.hid) - self.level
        return cooper.CMPState(loss=loss, ineq_defect=ineq, eq_defect=None)

    # ──────────────────────────────────────────────────────────────
    def metrics(self) -> tuple[float, float]:
        """(loss, ineq) as floats – call *after* the last closure."""
        state = self.last_cmp_state  # Cooper stores it here
        return float(state.loss), float(state.ineq_defect)


class Logger:
    "A thin, side-effect-free wrapper around wandb."

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

    def dumpkvs(self):
        if self._buf:
            self._run.log(self._buf)
            self._buf = {}


class LazyCPUData(Dataset):
    """
    Lazy CPU-side view of delta_z.
    No data lives on GPU until the DataLoader moves a batch.
    We keep tensors on the CPU to avoid OOM errors.
    """

    def __init__(self, h5_path: str, key: str = "cfc_train"):
        self.h5 = h5py.File(h5_path, "r")
        obj = self.h5[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(
                f"{key} is not an HDF5 dataset (found {type(obj)})"
            )
        self.ds: h5py.Dataset = obj

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        cfc = self.ds[idx]  # shape (2, rep_dim)
        delta_z = np.asarray(cfc[1] - cfc[0], dtype=np.float32)
        return torch.from_numpy(delta_z)

    def __del__(self):
        self.h5.close()


KEYS = ("batch", "lr", "oc", "seed", "norm")  # choose what matters


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
    form,
    cfg,
    dev,
) -> Tuple[float, float, float]:
    dict.train()
    recon_sum = 0.0
    sparsity_sum = 0.0
    concept_counts = 0
    for delta_z_cpu in dataloader:
        delta_z = delta_z_cpu.to(dev, non_blocking=True)
        delta_z, info = layer_normalise_(delta_z)
        optim.zero_grad()
        # ---------------------------------------------------------------
        # 1️⃣  Forward pass
        # ---------------------------------------------------------------
        lagrangian = form.composite_objective(
            ssae.closure, delta_z, info, cfg.loss
        )
        # ---------------------------------------------------------------
        # 2️⃣  Back-prop
        # ---------------------------------------------------------------
        form.custom_backward(lagrangian)  # custom backward

        # ---------------------------------------------------------------
        # 4️⃣  Optimiser step & scaler update
        # ---------------------------------------------------------------
        optim.step(ssae.closure, delta_z, info, cfg.loss)
        # ---------------------------------------------------------------
        # 5️⃣  Keep decoder columns unit normed
        # ---------------------------------------------------------------
        with torch.no_grad():
            renorm_decoder_cols_(ssae.model)
            project_decoder_grads_(ssae.model)
        # ---------------------------------------------------------------
        # 6️⃣  Log the loss and sparsity penalty
        # ---------------------------------------------------------------
        # Note: the loss is already scaled by the scaler, so no need to scale again
        # and the sparsity penalty is not scaled, so we can log it directly
        # Note: the lagrangian is a scalar, so we can log it directly
        recon_loss, sparsity_penalty = ssae.get_loss_values()
        recon_sum += recon_loss
        sparsity_sum += sparsity_penalty

        with torch.no_grad():  # cheap: only encoder
            concept_indicators = ssae.model.encode(delta_z)
            concept_counts += (
                (concept_indicators.abs() >= cfg.indicator_threshold)
                .sum()
                .item()
            )
        # ----------------------------------------------------------------
        # 7️⃣  Housekeeping: clear the graph and empty the cache
        # ----------------------------------------------------------------
        del lagrangian  # empty the graph
        del delta_z, concept_indicators  # clear unnecessary variables
        torch.cuda.empty_cache()  # clear cache to avoid OOM
    return recon_sum, sparsity_sum, concept_counts


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    logger = Logger(entity="causalrepl", project="ssae", config=asdict(cfg))
    dataloader = make_dataloader(cfg)
    dict = make_dict(cfg)
    ssae = make_ssae(dict, cfg)
    optim, form = make_optim(dict=dict, ssae=ssae, cfg=cfg)
    dev = next(dict.parameters()).device

    for ep in range(cfg.epochs):
        bs = len(dataloader)
        ds = len(dataloader.dataset)
        rec, sp, l0 = train_epoch(
            dataloader, dict, ssae, optim, form, cfg, dev
        )
        for k, v in {
            "recon": rec / bs,
            "spars": sp / bs,
            "l0": l0 / ds,
        }.items():
            logger.logkv(k, v)
        logger.dumpkvs()
        print(f"ep {ep:04d}  rec {rec:.4f}  spars {sp:.4f}")

    dump_run(cfg.emb.parent / "run_out", dict, cfg)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


# ===== run config ============================================================
@dataclass(frozen=True)
class Cfg:
    emb: Path
    data: Path
    batch: int = 32
    epochs: int = 20_000
    lr: float = 5e-4
    oc: int = 2
    n_concepts: int = 1
    norm: str = "ln"
    loss: str = "relative"
    ind_th: float = 0.1
    seed: int = 0
    rep_dim: int = 1_024  # read from data-cfg in real code

    # spill-over lives here (read-only)
    extra: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({}), init=False
    )

    @property
    def hid(self) -> int:  # encoder dim
        return self.oc * self.n_concepts


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("emb", type=Path)
    add("data-cfg", type=Path)
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=20_000)
    add("--lr", type=float, default=5e-4)
    add("--oc", type=int, default=2)
    add("--n-concepts", "-C", type=int, default=1)
    add("--warmup", type=int, default=2_000)
    add("--schedule", type=int, default=5_000)
    add("--target", type=float, default=0.1)
    add("--norm", choices=["ln", "gn", "bn"], default="ln")
    add("--loss", default="relative", choices=["relative", "absolute"])
    add("--ind-th", type=float, default=0.1)
    add("--seed", type=int, default=0)
    cli: Dict[str, Any] = vars(p.parse_args())

    yaml_path = cli.pop("data_cfg")
    yaml_cfg = _load_yaml(yaml_path)
    # ----- split YAML into (known, extra) -----------------------------------
    fields = {f.name for f in fields(Cfg)}
    extra = {k: v for k, v in yaml_cfg.items() if k not in fields}
    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", MappingProxyType(extra))
    return cfg


def make_dataloader(cfg) -> DataLoader:
    train_loader = DataLoader(
        LazyCPUData(cfg.emb, key="cfc_train"),
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=2,  # num_workers=0,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader


def make_dict(cfg: Cfg) -> torch.nn.Module:
    return DictLinearAE(cfg.rep_dim, cfg.hid, cfg.norm).cuda()


def make_ssae(model: torch.nn.Module, cfg: Cfg):
    return SSAE(
        model=model,
        batch=cfg.batch,
        hid=cfg.hid,
        n_concepts=cfg.n_concepts,
        warmup=cfg.warmup,
        schedule=cfg.schedule,
        target=cfg.target,
    )


def make_optim(dict: torch.nn.Module, ssae, cfg: Cfg):
    primal_optimizer = cooper.optim.ExtraAdam(
        list(dict.parameters()), lr=cfg.lr
    )
    dual_lr = cfg.lr / 2  # dual_lr should be less than primal_lr
    # 0.5 is common from extra gradient method literature
    dual_optimizer = cooper.optim.partial_optimizer(
        cooper.optim.ExtraAdam, lr=dual_lr
    )

    # Setup the constrained optimizer using Cooper's Lagrangian formulation
    formulation = cooper.LagrangianFormulation(ssae)
    coop_optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )
    return coop_optimizer, formulation


if __name__ == "__main__":
    with dbg.debug_on_exception():
        main()
