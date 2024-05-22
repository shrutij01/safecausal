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

import utils


class SparseDict(nn.Module):
    """x: differences between the embeddings of two contexts
    which differ in k-concepts
    M: rows of this are the sparse codes which express this diff
    c: ensures x is a sparse combination of M's rows
    """

    def __init__(self, embedding_size, overcomplete_basis_size):
        super(SparseDict, self).__init__()
        self.encoder = nn.Linear(
            embedding_size, overcomplete_basis_size, bias=True
        )
        self.decoder = nn.Linear(overcomplete_basis_size, embedding_size)

    def forward(self, x):
        c = torch.relu(self.encoder(x))  # this is ReLU(M.Tx + b)
        x_hat = self.decoder(c)  # this is Mc
        return x_hat, c


def train(dataloader, model, optimizer, loss_fxn, args):
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
                abs_loss = torch.abs(c).sum(dim=-1)
                l1_reg = abs_loss.sum() / x.shape[1]
                total_loss = reconstruction_error + float(args.alpha) * l1_reg

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        import ipdb

        ipdb.set_trace()
        x = (
            x_df_train[cfc_columns]
            .apply(lambda row: sum(row, []), axis=1)
            .tolist()
        )
    else:
        raise NotImplementedError

    import ipdb

    ipdb.set_trace()

    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    x = (x - mean) / (std + 1e-8)
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

    model = SparseDict(
        embedding_size=embedding_dim,
        overcomplete_basis_size=int(args.overcomplete_basis_factor)
        * embedding_dim,
    )
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-5
    )
    loss_fxn = torch.nn.MSELoss()
    losses = train(
        dataloader=loader,
        model=model,
        optimizer=optimizer,
        loss_fxn=loss_fxn,
        args=args,
    )
    modeldir = os.path.join(args.embedding_dir, "sparse_dict_model_", args.lr)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    else:
        overwrite = input(
            "A sparse dict model already exists at {}. Do you want to overwrite it? (yes/no): ".format(
                modeldir
            )
        )
        if overwrite.lower() != "yes":
            print("Skipping tf learner for now! Check trained model first!")
            exit()
    model_dict_path = os.path.join(modeldir, "model_M.pth")
    torch.save(model.state_dict(), model_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("--data-type", default="emb", choices=["emb", "gt"])
    parser.add_argument("--num-epochs", default=500)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--lr", type=float, default=float(1e-3))
    parser.add_argument("--alpha", type=float, default=float(1e-3))
    parser.add_argument("--overcomplete-basis-factor", type=int, default=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
