import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from typing import Callable, Any
import wandb
import h5py
import os
import yaml
from box import Box
import pandas as pd
import ast
import datetime
import argparse
from dataclasses import dataclass


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


class LinearDecayLR:
    def __init__(self, optimizer, total_steps, last_percentage=0.2):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.decay_start_step = int(total_steps * (1 - last_percentage))
        self.decay_steps = total_steps - self.decay_start_step
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step):
        if current_step < self.decay_start_step:
            # No decay needed before the last 20% of the steps
            lr = self.initial_lrs[0]
        else:
            # Calculate decayed lr
            decayed_lr = (
                (self.total_steps - current_step)
                / self.decay_steps
                * self.initial_lrs[0]
            )
            lr = max(decayed_lr, 0)  # Ensure lr does not go below 0

        # Apply the decayed learning rate
        for param_group, initial_lr in zip(
            self.optimizer.param_groups, self.initial_lrs
        ):
            param_group["lr"] = lr


class AlphaScheduler:
    def __init__(self, total_steps, max_value, increase_percentage=0.05):
        self.total_steps = total_steps
        self.max_value = max_value
        self.increase_end_step = int(total_steps * increase_percentage)
        self.alpha = 0  # Start at zero

    def get_coeff(self, current_step):
        if current_step <= self.increase_end_step:
            # Linearly increase
            self.alpha = (
                current_step / self.increase_end_step
            ) * self.max_value
        else:
            # Stay at max value
            self.alpha = self.max_value
        return self.alpha


class LinearSAE(nn.Module):
    """
    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        input_dim: int,
        rep_dim: int,
        activation: Callable = nn.ReLU(),
        normalize: bool = False,
    ) -> None:
        """
        :param input_dim: dimensionality of delta_z
        :param rep_dim: dimension of delta_c
        :param activation: activation function
        :param normalize: whether to preprocess data with layer norm
        """
        super(LinearSAE, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder: nn.Module = nn.Linear(input_dim, rep_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(rep_dim))
        self.activation = activation
        self.decoder = nn.Linear(rep_dim, input_dim, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor = torch.zeros(
            rep_dim, dtype=torch.long
        ).to(device)
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor

        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_(self)

    def encode_pre_act(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x,
            self.encoder.weight,
            self.latent_bias,
        )
        return latents_pre_act

    def preprocess(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = layer_normalise(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(
        self, latents: torch.Tensor, info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                autoencoder latents (shape: [batch, n_latents])
                reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return recons, latents


def unit_norm_decoder_(autoencoder: LinearSAE) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(
        dim=0
    )


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
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

    update_x(
        autoencoder.decoder.weight.grad,
        torch.einsum(
            "bn,bn->n",
            autoencoder.decoder.weight.data,
            autoencoder.decoder.weight.grad,
        ),
        autoencoder.decoder.weight.data,
        c=-1,
    )


class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, topk.values)
        return result


def sae_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    import ipdb

    ipdb.set_trace()
    recon_error = normalized_mean_squared_error(reconstruction, original_input)
    l0_error = non_normalized_L0_loss(latent_activations, original_input)
    l1_loss = normalized_L1_loss(latent_activations, original_input)
    sparsity_penalty = l1_weight * l1_loss
    total_loss = recon_error + sparsity_penalty
    return total_loss, recon_error, l0_error, sparsity_penalty


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1)
        / (original_input**2).mean(dim=1)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (
        latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)
    ).mean()


def non_normalized_L0_loss(
    latent_activations: torch.Tensor, original_input: torch.Tensor
) -> torch.Tensor:
    return (latent_activations.ne(0).sum(dim=1).float()).mean()


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


def load_training_data(args):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(
        args.embedding_dir,
        str(args.k) + str(args.seed) + timestamp_str,
    )
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    x = None
    if args.data_type == "emb":
        embeddings_file = os.path.join(args.embedding_dir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_train = np.array(f["cfc1_train"]).squeeze()
            cfc2_train = np.array(f["cfc2_train"]).squeeze()

        x = cfc2_train - cfc1_train
    elif args.data_type == "gt" or args.data_type == "gt_ent":
        config_file = os.path.join(args.embedding_dir, "config.yaml")
        with open(config_file, "r") as file:
            config = Box(yaml.safe_load(file))
        df_file = os.path.join(
            args.embedding_dir,
            "object_translations" + str(config.dgp) + ".csv",
        )

        cfc_columns = config.cfc_column_names
        converters = {col: ast.literal_eval for col in cfc_columns}
        x_df = pd.read_csv(df_file, converters=converters)
        x_df_train = x_df.iloc[0 : int(config.split * config.size)]

        def convert_to_list_of_ints(value):
            if isinstance(value, str):
                value = ast.literal_eval(value)
            return [int(x) for x in value]

        for column in x_df_train[cfc_columns]:
            x_df_train[column] = x_df_train[column].apply(
                convert_to_list_of_ints
            )

        x_gt = np.asarray(
            (
                x_df_train[cfc_columns]
                .apply(lambda row: sum(row, []), axis=1)
                .tolist()
            )
        )
        lin_ent_dim = x_gt.shape[1]

        def generate_invertible_matrix(size):
            while True:
                matrix = np.random.randint(1, size, (size, size))
                if np.linalg.det(matrix) != 0:
                    return matrix

        import ipdb

        ipdb.set_trace()
        lin_ent_tf = generate_invertible_matrix(lin_ent_dim)
        x_ent = np.array([lin_ent_tf @ x_gt[i] for i in range(x_gt.shape[0])])
        if args.data_type == "gt_ent":
            x = x_ent
            np.save(os.path.join(modeldir, "lin_ent_tf.npy"), lin_ent_tf)

    else:
        raise NotImplementedError

    embedding_dim = x.shape[1]

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    dataset = TensorDataset(x)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    return loader, embedding_dim, modeldir


def train(sae, dataloader, logger, num_epochs, lr, eps=6.25e-10):
    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx_manager = torch.cuda.amp.autocast()

    opt = torch.optim.Adam(sae.parameters(), lr=lr, eps=eps, fused=True)
    optim_scheduler = LinearDecayLR(opt, args.num_epochs)
    alpha_scheduler = AlphaScheduler(args.num_epochs, args.alpha)
    for epoch in range(int(num_epochs)):
        epoch_loss = 0.0
        l0_delta_c_hat = 0.0
        recon = 0.0
        sparsity_penalty = 0.0
        for delta_z_list in dataloader:
            assert len(delta_z_list) == 1
            delta_z = delta_z_list[0]
            assert isinstance(delta_z, torch.Tensor)
            with autocast_ctx_manager:
                import ipdb

                ipdb.set_trace()
                delta_z_hat, delta_c_hat = sae(delta_z)
                alpha = alpha_scheduler.get_coeff(epoch)
                loss, recon_error, l0, l1 = sae_loss(
                    delta_z_hat, delta_z, delta_c_hat, alpha
                )

            print(epoch, loss)
            epoch_loss += loss.item()
            l0_delta_c_hat += l0.item()
            recon += recon_error.item()
            sparsity_penalty += l1.item()
            logger.logkv("loss_scale", scaler.get_scale())
            loss = scaler.scale(loss)
            loss.backward()

            unit_norm_decoder_(sae)
            # unit_norm_decoder_grad_adjustment_(sae)
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            optim_scheduler.step(epoch)
        logger.logkv("total_loss", epoch_loss)
        logger.logkv("recon_error", recon)
        logger.logkv("l0_delta_c_hat", l0_delta_c_hat)
        logger.logkv("sparsity_penalty", sparsity_penalty)
        logger.dumpkvs()


def save(sae, config_dict, modeldir):
    sae_dir = os.path.join(modeldir, "sparse_dict_model")
    if not os.path.exists(sae_dir):
        os.makedirs(sae_dir)
    config_file = os.path.join(modeldir, "models_config.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config_dict, file)
    sae_dict_path = os.path.join(sae_dir, "sparse_dict_model.pth")
    torch.save(sae.state_dict(), sae_dict_path)


def main(args):
    config_dict = {
        "seed": args.seed,
        "dataset": args.embedding_dir,
        "data_type": args.data_type,
        "alpha": args.alpha,
        "k": args.k,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
    }
    set_seeds(int(args.seed))
    logger = Logger(project="psp", config=config_dict)
    dataloader, input_dim, modeldir = load_training_data(args)
    topk = TopK(k=args.k)
    sae = LinearSAE(
        input_dim=input_dim,
        rep_dim=input_dim,
        activation=topk,
        normalize=True,
    )
    sae.cuda()
    train(
        sae=sae,
        dataloader=dataloader,
        logger=logger,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )
    save(sae, config_dict, modeldir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_dir")
    parser.add_argument(
        "--data-type", default="emb", choices=["emb", "gt", "gt_ent"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=float(0.00))
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument(
        "--normalize",
        default=True,
    )
    parser.add_argument("--seed", default=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args, unknown = parser.parse_known_args()

    main(args)
