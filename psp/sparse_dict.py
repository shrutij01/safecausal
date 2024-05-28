import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

import argparse
import os
import h5py
import yaml
from box import Box
import numpy as np
import pandas as pd
import ast
import datetime

import utils

"""Implementation tricks from 1. https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
and 2. https://transformer-circuits.pub/2024/april-update/index.html#training-saes.
2. over 1. in case of contradictions.
"""


class SparseDict(nn.Module):
    """x: differences between the embeddings of two contexts
    which differ in k-concepts
    M: rows of this are the sparse codes which express this diff
    c: ensures x is a sparse combination of M's rows
    """

    def __init__(self, embedding_size, overcomplete_basis_size):
        super(SparseDict, self).__init__()
        self.encoder = nn.Linear(
            embedding_size, overcomplete_basis_size, bias=False
        )
        self.decoder = nn.Linear(
            overcomplete_basis_size, embedding_size, bias=False
        )

        self.bias_encoder = nn.Parameter(torch.zeros(overcomplete_basis_size))
        self.bias_decoder = nn.Parameter(torch.zeros(embedding_size))

        Wd_initial = torch.randn(embedding_size, overcomplete_basis_size)
        norms = torch.sqrt(torch.sum(Wd_initial**2, dim=0))
        desired_norms = torch.rand(overcomplete_basis_size) * 0.95 + 0.05
        scale_factors = desired_norms / norms
        self.decoder.weight = nn.Parameter(Wd_initial * scale_factors)

        self.encoder.weight = nn.Parameter(
            self.decoder.weight.detach().clone().t()
        )

    def forward(self, x):
        c = torch.relu(self.encoder(x))  # this is ReLU(W_e.Tx + b)
        x_hat = self.decoder(c)  # this is W_dc
        return x_hat, c


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


def train(
    dataloader,
    model,
    optimizer,
    optim_scheduler,
    alpha_scheduler,
    loss_fxn,
    args,
):
    scaler = GradScaler()
    losses = []
    for epoch in range(int(args.num_epochs)):
        epoch_loss = 0.0
        for x_list in dataloader:
            optimizer.zero_grad()
            with autocast():  # Enables mixed precision
                x = x_list[0]
                # todo: check why this appears as a list
                x_hat, c = model(x)
                reconstruction_error = loss_fxn(x_hat, x)

                sparsity_penalty = torch.sum(
                    torch.norm(c, p=1, dim=1).unsqueeze(dim=1)
                    * torch.norm(model.decoder.weight, p=2, dim=1).unsqueeze(
                        dim=0
                    ),
                )  # sparisty penalty on the columns
                alpha = alpha_scheduler.get_coeff(epoch)
                total_loss = reconstruction_error + alpha * sparsity_penalty

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optim_scheduler.step(epoch)
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=1
            )

            epoch_loss += total_loss.item()
        average_loss = epoch_loss / len(dataloader)
        losses.append(average_loss)
        if epoch % 100 == 0:
            print(f"Ending epoch {epoch}, Average Loss: {average_loss:.4f}")

    return losses


def main(args, device):
    if args.data_type == "emb":
        embeddings_file = os.path.join(args.embedding_dir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_train = np.array(f["cfc1_train"]).squeeze()
            cfc2_train = np.array(f["cfc2_train"]).squeeze()
        x = cfc2_train - cfc1_train
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
        x = (x - mean) / (std + 1e-8)
    elif args.data_type == "gt":
        df_file = os.path.join(
            args.embedding_dir, "multi_objects_single_coordinate.csv"
        )

        config_file = os.path.join(args.embedding_dir, "config.yaml")
        with open(config_file, "r") as file:
            config = Box(yaml.safe_load(file))
        cfc_columns = config.cfc_column_names
        converters = {col: ast.literal_eval for col in cfc_columns}
        x_df = pd.read_csv(df_file, converters=converters)
        x_df_train = x_df.iloc[: int(config.split * config.size)]

        def convert_to_list_of_ints(value):
            if isinstance(value, str):
                value = ast.literal_eval(value)
            return [int(x) for x in value]

        for column in x_df_train[cfc_columns]:
            x_df_train[column] = x_df_train[column].apply(
                convert_to_list_of_ints
            )

        x = np.asarray(
            (
                x_df_train[cfc_columns]
                .apply(lambda row: sum(row, []), axis=1)
                .tolist()
            )
        )
    else:
        raise NotImplementedError

    embedding_dim = x.shape[1]

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    dataset = TensorDataset(x)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    overcomplete_basis_size = (
        int(args.overcomplete_basis_factor) * embedding_dim
    )
    model = SparseDict(
        embedding_size=embedding_dim,
        overcomplete_basis_size=overcomplete_basis_size,
    )
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-5
    )
    optim_scheduler = LinearDecayLR(optimizer, args.num_epochs)
    alpha_scheduler = AlphaScheduler(args.num_epochs, args.alpha)

    loss_fxn = torch.nn.MSELoss()
    losses = train(
        dataloader=loader,
        model=model,
        optimizer=optimizer,
        optim_scheduler=optim_scheduler,
        alpha_scheduler=alpha_scheduler,
        loss_fxn=loss_fxn,
        args=args,
    )
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(
        args.embedding_dir, "sparse_dict_model", timestamp_str
    )
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    model_config_file = os.path.join(modeldir, "model_config.yaml")
    model_config = {
        "embedding_size": embedding_dim,
        "overcomplete_basis_size": overcomplete_basis_size,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "data_type": args.data_type,
    }
    with open(model_config_file, "w") as file:
        yaml.dump(model_config, file)
    model_dict_path = os.path.join(modeldir, "model_M.pth")
    torch.save(model.state_dict(), model_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("--data-type", default="emb", choices=["emb", "gt"])
    parser.add_argument("--num-epochs", default=500)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--lr", type=float, default=float(5e-5))
    parser.add_argument("--alpha", type=float, default=float(5))
    parser.add_argument("--overcomplete-basis-factor", type=int, default=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
