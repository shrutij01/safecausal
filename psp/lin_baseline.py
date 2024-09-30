import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import cooper

import numpy as np

from typing import Callable, Any
import wandb
import h5py
import os
import yaml
import pickle
from box import Box
import pandas as pd
import ast
import datetime
import argparse
from psp.data_utils import tensorify, numpify
from psp.metrics import mean_corr_coef


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seeds(seed):
    """Set seeds for reproducibility."""
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Seeds the GPU if available
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


def layer_normalise(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


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
        num_concepts: int,
        norm_type: str,
    ) -> None:
        super(LinearSAE, self).__init__()
        self.encoder: nn.Module = nn.Linear(
            rep_dim, num_concepts, bias=False, device=device
        )
        if norm_type == "ln":
            self.encoder_ln = nn.LayerNorm(num_concepts, device=device)
        elif norm_type == "gn":
            self.encoder_ln = nn.GroupNorm(1, num_concepts, device=device)
        elif norm_type == "bn":
            self.encoder_ln = nn.BatchNorm1d(num_concepts, device=device)
        else:
            raise ValueError("Invalid norm type, pass ln, gn, or bn")
        self.encoder_bias = nn.Parameter(
            torch.zeros(num_concepts, device=device)
        )
        self.decoder = nn.Linear(
            num_concepts, rep_dim, bias=False, device=device
        )
        self.act = nn.LeakyReLU(0.1)
        self.decoder_bias = nn.Parameter(torch.zeros(rep_dim, device=device))
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_columns(self)

    def preprocess(
        self, delta_z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        delta_z, mu, std = layer_normalise(delta_z)
        return delta_z, dict(mu=mu, std=std)

    def forward(
        self, delta_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param delta_z: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                autoencoder latents (shape: [batch, n_latents])
                reconstructed data (shape: [batch, n_inputs])
        """
        delta_z = delta_z.to(device)
        delta_z, info = self.preprocess(delta_z)

        concept_indicators = (
            self.encoder_ln((self.encoder(delta_z - self.decoder_bias)))
            + self.encoder_bias
        )
        delta_z_hat = self.decoder(concept_indicators) + self.decoder_bias
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
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    def update_x(x, a, b, c):
        # Ensure all tensors are on the same device
        if not x.is_cuda:
            x = x.cuda()

        if not a.is_cuda:
            a = a.cuda()

        if not b.is_cuda:
            b = b.cuda()

        # Perform the operation
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


def compute_loss(delta_z, delt_z_hat):
    return (
        ((delt_z_hat - delta_z) ** 2).mean(dim=1) / (delta_z**2).mean(dim=1)
    ).mean()


class Logger:
    def __init__(self, **kws):
        self.vals = {}
        wandb.init(**kws)

    def logkv(self, k, v):
        self.vals[k] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        wandb.log(self.vals)
        self.vals = {}


def load_training_data(args, config) -> tuple[DataLoader, int, int]:
    delta_z_train = None
    rep_dim = config.rep_dim
    num_concepts = config.num_concepts
    if (
        config.dataset == "binary_1"
        or config.dataset == "binary_1_2"
        or config.dataset == "binary_1_1"
        or config.dataset == "binary_2"
        or config.dataset == "truthful_qa"
        or config.dataset == "categorical"
    ):
        with h5py.File(args.embeddings_file, "r") as f:
            cfc_train = np.array(f["cfc_train"]).squeeze()
        delta_z_train = tensorify(cfc_train[:, 1] - cfc_train[:, 0], device)
    else:
        raise NotImplementedError
    train_dataset = TensorDataset(
        delta_z_train,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    return train_loader, rep_dim, num_concepts


def save(args, sae_model, config_dict):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(
        os.path.dirname(args.embeddings_file),
        "lin_baseline_"
        + str(args.primal_lr)
        + "_"
        + str(config_dict["llm_layer"])
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


def train(
    train_loader,
    sae_model,
    optimizer,
    logger,
    num_epochs,
):
    for epoch in range(int(num_epochs)):
        epoch_loss = 0.0
        sparsity_penalty_total = 0.0
        for delta_z_list in train_loader:
            delta_z = delta_z_list[0]
            # this makes bn use batch statistics while training, doesn't have any
            # effect for gn or ln
            sae_model.train()
            optimizer.zero_grad()
            delta_z_hat, _ = sae_model(delta_z)
            total_loss = compute_loss(delta_z_hat, delta_z)

            total_loss.backward()

            optimizer.step()

            # Normalize the decoder columns to unit length after the parameters update
            unit_norm_decoder_columns(sae_model)

            # Adjust the gradients post-optimization step to maintain unit norms
            unit_norm_decoder_columns_grad_adjustment_(sae_model)

            epoch_loss += compute_loss(delta_z_hat, delta_z).item()
        logger.logkv("total_loss", epoch_loss)
        logger.logkv("sparsity_penalty", sparsity_penalty_total)
        print(epoch, epoch_loss)
        logger.dumpkvs()


def main(args):
    with open(args.data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    config_dict = {
        "seed": args.seed,
        "embeddings_file": args.embeddings_file,
        "lr": args.primal_lr,
        "norm_type": args.norm_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "indicator_threshold": args.indicator_threshold,
        "llm_layer": data_config.llm_layer,
    }
    set_seeds(int(args.seed))
    logger = Logger(project="iclrpsp", config=config_dict)
    train_loader, rep_dim, num_concepts = load_training_data(args, data_config)
    # Assuming the LinearSAE model and other parameters are already defined:
    sae_model = LinearSAE(
        rep_dim=rep_dim, num_concepts=num_concepts, norm_type=args.norm_type
    )
    optimizer = torch.optim.AdamW(
        sae_model.parameters(), lr=float(args.primal_lr), weight_decay=1e-5
    )

    train(
        train_loader=train_loader,
        sae_model=sae_model,
        optimizer=optimizer,
        logger=logger,
        num_epochs=args.num_epochs,
    )
    save(sae_model=sae_model, args=args, config_dict=config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file")
    parser.add_argument("--data-config-file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20000)
    parser.add_argument("--primal-lr", type=float, default=0.01)
    parser.add_argument("--indicator-threshold", type=float, default=0.1)
    # indicator threshold needs to be decently low so that the concept_indicators
    # don't turn out to be all zeros
    parser.add_argument("--num-concepts", type=int, default=3)
    parser.add_argument(
        "--norm-type", default="bn", choices=["ln", "gn", "bn"]
    )
    parser.add_argument("--seed", default=0)

    args, unknown = parser.parse_known_args()

    main(args)
