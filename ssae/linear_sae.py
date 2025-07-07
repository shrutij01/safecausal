import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cooper
import math

import numpy as np

from typing import Any
import wandb
import h5py
import os
import yaml
from box import Box
import pandas as pd
import ast
import datetime
import argparse
from ssae.data_utils import tensorify
import debug_tools as dbg

import os

from IPython.core.debugger import Pdb
from contextlib import contextmanager


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seeds(seed):
    """Set seeds for reproducibility."""
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Seeds the GPU if available
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


def layer_normalise(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # in-place operations indicated by _ suffix cuz non-normalized x
    # is not needed
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(-1, keepdim=True).clamp_min_(eps)
    x.sub_(mu).div_(std)
    return x, dict(mu=mu, std=std)


class LinearSAE(nn.Module):
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
        encoder_dim: int,
        norm_type: str,
    ) -> None:
        super(LinearSAE, self).__init__()
        self.encoder: nn.Module = nn.Linear(
            rep_dim, encoder_dim, bias=False, device=device
        )
        if norm_type == "ln":
            self.encoder_ln = nn.LayerNorm(encoder_dim, device=device)
        elif norm_type == "gn":
            self.encoder_ln = nn.GroupNorm(1, encoder_dim, device=device)
        elif norm_type == "bn":
            self.encoder_ln = nn.BatchNorm1d(encoder_dim, device=device)
            # really important mention in apx details
        else:
            raise ValueError("Invalid norm type, pass ln, gn, or bn")
        self.encoder_bias = nn.Parameter(
            torch.zeros(encoder_dim, device=device)
        )
        # Bias (approach from nn.Linear)
        fan_in = self.encoder.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform(self.encoder_bias, -bound, bound)
        # nn.init.xavier_uniform_(self.encoder.weight)  # diff

        self.decoder = nn.Linear(
            encoder_dim, rep_dim, bias=False, device=device
        )
        self.decoder_bias = nn.Parameter(torch.zeros(rep_dim, device=device))
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_columns(self)
        # no unit norm init here for decoder # diff

    def preprocess(
        self, delta_z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        delta_z, info = layer_normalise(delta_z)
        return delta_z, info

    def encode(self, delta_z: torch.Tensor) -> torch.Tensor:
        """
        expects delta_z to be already normalised
        :param delta_z: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        delta_z = delta_z.to(device)
        # delta_z, info = self.preprocess(delta_z)
        concept_indicators = self.encoder(delta_z - self.decoder_bias)  # diff
        concept_indicators = self.encoder_ln(concept_indicators)
        concept_indicators += self.encoder_bias
        return concept_indicators

    def forward(
        self, delta_z: torch.Tensor, info: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param delta_z: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                autoencoder latents (shape: [batch, n_latents])
                reconstructed data (shape: [batch, n_inputs])
        """
        concept_indicators = self.encode(delta_z)
        delta_z_hat = (
            self.decoder(concept_indicators) + self.decoder_bias
        )  # diff
        delta_z_hat = delta_z_hat * info["std"] + info["mu"]

        return delta_z_hat, concept_indicators


def unit_norm_decoder_columns(autoencoder: LinearSAE) -> None:
    """
    Unit normalize the columns of the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(
        dim=0, keepdim=True
    )


def unit_norm_decoder_columns_grad_adjustment_(autoencoder) -> None:
    """project out gradient information parallel to the dictionary vectors
    - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    def update_x(x, a, b, c):
        if not x.is_cuda:
            x = x.cuda()
        if not a.is_cuda:
            a = a.cuda()
        if not b.is_cuda:
            b = b.cuda()
        x += a * b * c
        return x

    # Calculate the gradient adjustment for each column of the decoder
    for i in range(autoencoder.decoder.weight.shape[1]):
        column = autoencoder.decoder.weight[:, i]
        grad = autoencoder.decoder.weight.grad[:, i]
        update_x(
            grad,
            torch.einsum("b,b->", column, grad),
            column,
            c=-1,
        )


class ConstrainedLinearSAE(cooper.ConstrainedMinimizationProblem):
    def __init__(
        self,
        model,
        overcompleteness_factor,
        batch_size,
        encoder_dim,
        num_concepts,
        warmup_epochs: int = 2000,
        total_schedule_epochs: int = 5000,
        target_sparse_level: float = 5,
        sparsity_factor: int = 5,
    ):
        # here initial sparse level > target sparse level
        super().__init__(is_constrained=True)
        self.model = model
        self.criterion = nn.MSELoss(reduction="mean")
        assert sparsity_factor >= 1
        self.target_sparse_level = target_sparse_level  # diff
        self.initial_sparse_level = num_concepts
        self.sparse_level = num_concepts
        self.batch_size = batch_size
        self.encoder_dim = encoder_dim
        self.num_concepts = num_concepts
        self.warmup_epochs = warmup_epochs
        self.total_schedule_epochs = total_schedule_epochs
        self.current_epoch = 0

    def compute_loss(self, delta_z, delta_z_hat, type="relative"):
        if type == "relative":
            return (
                ((delta_z - delta_z_hat) ** 2).mean(dim=1)
                / (delta_z**2).mean(dim=1)
            ).mean()
        elif type == "absolute":
            return (((delta_z - delta_z_hat) ** 2).mean(dim=1)).mean()  # diff
        else:
            raise ValueError

    def step_scheduler(self):
        """
        Linearly decrease sparse_level (enforce stricter sparsity) from initial to target after warmup.
        """
        if self.current_epoch < self.warmup_epochs:
            return  # no update

        relative_epoch = self.current_epoch - self.warmup_epochs
        if self.total_schedule_epochs == 0:
            ratio = 1.0
        else:
            ratio = min(1.0, relative_epoch / self.total_schedule_epochs)
        self.sparse_level = self.initial_sparse_level - ratio * (
            self.initial_sparse_level - self.target_sparse_level
        )  # diff

    def closure(self, delta_z, info, loss_type="relative"):
        """
        This closure function computes the model's loss and the constraints.
        :param inputs: Input tensor to the model
        :param targets: Target tensor for reconstruction comparison
        """
        delta_z_hat, concept_indicators = self.model(delta_z, info)
        self.loss = self.compute_loss(delta_z, delta_z_hat, loss_type)
        # Compute the sparsity constraint as an inequality defect
        self.step_scheduler()  # diff missing piece
        self.ineq_defect = (
            torch.sum(torch.abs(concept_indicators))
            / self.batch_size
            / self.encoder_dim
            - self.sparse_level
        )
        return cooper.CMPState(
            loss=self.loss, ineq_defect=self.ineq_defect, eq_defect=None
        )

    def get_loss_values(self):
        return self.loss.item(), self.ineq_defect.item()


class Logger:
    def __init__(self, **kws):
        self.vals = {}
        os.environ["WANDB_DIR"] = "/network/scratch/j/joshi.shruti/wandb_logs"
        wandb.init(
            dir="/network/scratch/j/joshi.shruti/wandb_logs",  # optional if WANDB_DIR is set
            settings=wandb.Settings(code_dir="."),  # no code copying
            **kws,
        )

    def logkv(self, k, v):
        self.vals[k] = float(v) if torch.is_tensor(v) else v
        # v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        wandb.log(self.vals)
        self.vals = {}


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


def load_training_data(args) -> DataLoader:
    train_loader = DataLoader(
        LazyCPUData(args.embeddings_file, key="cfc_train"),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=2,  # num_workers=0,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader


def save(args, sae_model, config_dict):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(
        os.path.dirname(args.embeddings_file),
        str(config_dict["llm_layer"])
        + "_"
        + str(config_dict["loss_type"])
        + "_"
        + str(args.warmup_epochs)
        + "_"
        + str(args.scheduler_epochs)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.overcompleteness_factor)
        + "_"
        + str(args.norm_type)
        + "_"
        + str(args.primal_lr)
        + "_"
        + str(args.seed),
    )
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    config_file = os.path.join(
        modeldir,
        "model_config.yaml",
    )
    with open(config_file, "w") as file:
        yaml.dump(config_dict, file)
    sae_dict_path = os.path.join(modeldir, "sparse_dict_model.pth")
    torch.save(sae_model.state_dict(), sae_dict_path)


def to_device(data, device):
    # to handle oom error
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def train(
    train_loader,
    sae_model,
    cmp_model,
    coop_optimizer,
    formulation,
    logger,
    args,
) -> None:
    scaler = GradScaler(enabled=torch.cuda.is_available())
    num_batches = len(train_loader)
    dataset_size = len(train_loader.dataset)
    for epoch in range(int(args.num_epochs)):
        sae_model.train()
        recon_sum = 0.0
        sparsity_sum = 0.0
        concept_counts = 0
        for delta_z_cpu in train_loader:
            delta_z = delta_z_cpu.to(device, non_blocking=True)
            delta_z, info = layer_normalise(delta_z)
            coop_optimizer.zero_grad()
            # ---------------------------------------------------------------
            # 1️⃣  Forward pass under autocast
            # ---------------------------------------------------------------
            with autocast(enabled=torch.cuda.is_available()):
                lagrangian = formulation.composite_objective(
                    cmp_model.closure, delta_z, info, args.loss_type
                )
            # ---------------------------------------------------------------
            # 2️⃣  Back-prop — scale the loss manually
            # ---------------------------------------------------------------
            scaled_loss = scaler.scale(lagrangian)  # multiply by scale factor
            formulation.custom_backward(scaled_loss)  # custom backward

            # ---------------------------------------------------------------
            # 3️⃣  Un-scale the gradients **in place**
            # ---------------------------------------------------------------
            scaler.unscale_(coop_optimizer.primal_optimizer)
            # ---------------------------------------------------------------
            # 4️⃣  Optimiser step & scaler update
            # ---------------------------------------------------------------
            coop_optimizer.step(
                cmp_model.closure, delta_z, info, args.loss_type
            )
            scaler.update()  # update the scale factor

            # ---------------------------------------------------------------
            # 5️⃣  Keep decoder columns unit normed
            # ---------------------------------------------------------------
            with torch.no_grad():
                unit_norm_decoder_columns(cmp_model.model)
                unit_norm_decoder_columns_grad_adjustment_(cmp_model.model)
            # ---------------------------------------------------------------
            # 6️⃣  Log the loss and sparsity penalty
            # ---------------------------------------------------------------
            # Note: the loss is already scaled by the scaler, so no need to scale again
            # and the sparsity penalty is not scaled, so we can log it directly
            # Note: the lagrangian is a scalar, so we can log it directly
            recon_loss, sparsity_penalty = cmp_model.get_loss_values()
            recon_sum += recon_loss
            sparsity_sum += sparsity_penalty

            with torch.no_grad():  # cheap: only encoder
                concept_indicators = cmp_model.model.encode(delta_z)
                concept_counts += (
                    (concept_indicators.abs() >= args.indicator_threshold)
                    .sum()
                    .item()
                )
            # ----------------------------------------------------------------
            # 7️⃣  Housekeeping: clear the graph and empty the cache
            # ----------------------------------------------------------------
            del lagrangian  # empty the graph
            del delta_z, concept_indicators  # clear unnecessary variables
            torch.cuda.empty_cache()  # clear cache to avoid OOM
        logger.logkv("total_loss", recon_sum / num_batches)
        logger.logkv("sparsity_penalty", sparsity_sum / num_batches)
        logger.logkv(
            "num_concepts_predicted",
            concept_counts / dataset_size,
        )
        logger.dumpkvs()
        print(
            f"epoch {epoch:04d} | "
            f"recon {recon_sum / num_batches:.5f} | "
            f"sparsity {sparsity_sum / num_batches:.5f}"
        )


def main(args):
    with open(args.data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    config_dict = {
        "seed": args.seed,
        "embeddings_file": args.embeddings_file,
        "overcompleteness_factor": args.overcompleteness_factor,
        "primal_lr": args.primal_lr,
        "dual_lr": args.primal_lr / 2,
        "num_concepts": args.num_concepts,
        "norm_type": args.norm_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "indicator_threshold": args.indicator_threshold,
        "loss_type": args.loss_type,
        "warmup_epochs": args.warmup_epochs,
        "scheduler_epochs": args.scheduler_epochs,
        "sparsity_factor": args.sparsity_factor,
        "target_sparse_level": args.target_sparse_level,
        "dataset": data_config.dataset,
        "rep_dim": data_config.rep_dim,
        "llm_layer": data_config.llm_layer,
    }
    set_seeds(int(args.seed))
    logger = Logger(entity="causalrepl", project="ssae", config=config_dict)
    train_loader = load_training_data(args)
    # Assuming the LinearSAE model and other parameters are already defined:
    encoder_dim = int(args.overcompleteness_factor) * int(args.num_concepts)
    rep_dim = int(data_config.rep_dim)
    assert encoder_dim <= rep_dim
    sae_model = LinearSAE(
        rep_dim=rep_dim, encoder_dim=encoder_dim, norm_type=args.norm_type
    )
    total_schedule_epochs = min(
        args.scheduler_epochs, args.num_epochs - args.warmup_epochs
    )
    cmp_model = ConstrainedLinearSAE(
        model=sae_model,
        overcompleteness_factor=int(args.overcompleteness_factor),
        batch_size=int(args.batch_size),
        encoder_dim=encoder_dim,
        num_concepts=int(args.num_concepts),
        warmup_epochs=args.warmup_epochs,
        total_schedule_epochs=total_schedule_epochs,
        target_sparse_level=args.target_sparse_level,
        sparsity_factor=args.sparsity_factor,
    )

    # Define optimizers
    primal_optimizer = cooper.optim.ExtraAdam(
        list(sae_model.parameters()), lr=args.primal_lr
    )
    dual_lr = args.primal_lr / 2  # dual_lr should be less than primal_lr
    # 0.5 is common from extra gradient method literature
    dual_optimizer = cooper.optim.partial_optimizer(
        cooper.optim.ExtraAdam, lr=dual_lr
    )

    # Setup the constrained optimizer using Cooper's Lagrangian formulation
    formulation = cooper.LagrangianFormulation(cmp_model)
    coop_optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    train(
        train_loader=train_loader,
        sae_model=sae_model,
        cmp_model=cmp_model,
        coop_optimizer=coop_optimizer,
        formulation=formulation,
        logger=logger,
        args=args,
    )
    save(sae_model=sae_model, args=args, config_dict=config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file")
    parser.add_argument("--data-config-file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20000)
    parser.add_argument("--primal-lr", type=float, default=0.0005)
    parser.add_argument("--dual-lr", type=float, default=0.005)
    # leaving this here for now but changing it to primal_lr/2 in the code
    # following the partial observability paper and repo
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--scheduler-epochs", type=int, default=500)
    parser.add_argument("--sparsity-factor", type=float, default=5)
    parser.add_argument("--target-sparse-level", type=float, default=0.1)
    # say it starts at num_concepts and then goes down
    parser.add_argument("--overcompleteness-factor", type=int, default=2)
    parser.add_argument("--indicator-threshold", type=float, default=0.1)
    # indicator threshold needs to be decently low so that the concept_indicators
    # don't turn out to be all zeros
    parser.add_argument("--num-concepts", type=int, default=1)
    parser.add_argument(
        "--norm-type", default="ln", choices=["ln", "gn", "bn"]
    )
    parser.add_argument(
        "--loss-type", default="relative", choices=["relative", "absolute"]
    )
    parser.add_argument("--seed", default=0)

    args, unknown = parser.parse_known_args()
    with dbg.debug_on_exception():
        main(args)
