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
from terminalplot import plot
from psp.utils import DisentanglementScores


"""Implementation tricks from 1. https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
and 2. https://transformer-circuits.pub/2024/april-update/index.html#training-saes.
2. over 1. in case of contradictions.
"""


class LinearInvertible(nn.Module):
    # explicitly demands WW.T = I, the non-degeneracy condition in linear ICA
    # objective to train is just the sparsity penalty for this
    def __init__(self, embedding_size, overcomplete_basis_size):
        super(LinearInvertible, self).__init__()
        self.encoder = nn.Linear(
            embedding_size, overcomplete_basis_size, bias=False
        )
        self.decoder = nn.Linear(
            overcomplete_basis_size, embedding_size, bias=False
        )
        nn.init.orthogonal_(self.encoder.weight)
        self.encoder.weight = nn.Parameter(
            self.decoder.weight.detach().clone().t()
        )  # transpose of an orthogonal matrix is also orthogonal

    def forward(self, delta_z):
        def whiten(x):
            x = x.detach().cpu()
            mean = torch.mean(x, dim=0, keepdim=True)
            x_centered = x - mean

            # Step 2: Compute covariance matrix
            cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)

            # Step 3: Eigenvalue decomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Step 4: Compute the transformation matrix
            # Adding a small constant to avoid division by zero in case of zero eigenvalues
            epsilon = 1e-5
            transformation_matrix = eigenvectors / torch.sqrt(
                eigenvalues + epsilon
            ).unsqueeze(0)

            # Step 5: Transform the original matrix
            x_whitened = x_centered @ transformation_matrix
            x_whitened = x_whitened.to(device).type(torch.float32)
            return x_whitened

        delta_z = whiten(delta_z)
        delta_c = self.encoder(delta_z)  # this is W_e.Tx + b_e
        r_delta_z_hat = self.decoder(delta_c)  # this is W_d.c + b_d
        return r_delta_z_hat, delta_c


class LinearAutoencoder(nn.Module):
    # based on RICA, non-degeneracy is handled by the recon error
    """x: differences between the embeddings of two contexts
    which differ in k-concepts
    M: rows of this are the sparse codes which express this diff
    c: ensures x is a sparse combination of M's rows
    this iteration also has updates from anthropic's implementation
    """

    def __init__(self, embedding_size, overcomplete_basis_size):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(
            embedding_size, overcomplete_basis_size, bias=False
        )
        self.decoder = nn.Linear(
            overcomplete_basis_size, embedding_size, bias=False
        )

        self.bias_encoder = nn.Parameter(torch.zeros(overcomplete_basis_size))
        self.bias_decoder = nn.Parameter(torch.zeros(embedding_size))

        self.encoder.weight = nn.Parameter(
            self.decoder.weight.detach().clone().t()
        )

    def forward(self, r_delta_z):
        delta_c = (
            self.encoder(r_delta_z) + self.bias_encoder
        )  # this is W_e.Tx + b_e
        r_delta_z_hat = (
            self.decoder(delta_c) + self.bias_decoder
        )  # this is W_d.c + b_d
        return r_delta_z_hat, delta_c


class NonLinearAutoencoder(nn.Module):
    def __init__(self, embedding_size, overcomplete_basis_size, act_fxn):
        super(NonLinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(
            embedding_size, overcomplete_basis_size, bias=False
        )
        self.decoder = nn.Linear(
            overcomplete_basis_size, embedding_size, bias=False
        )
        self.bias_encoder = nn.Parameter(torch.zeros(overcomplete_basis_size))
        self.bias_decoder = nn.Parameter(torch.zeros(embedding_size))
        if act_fxn == "leakyrelu":
            self.act_fxn = nn.LeakyReLU(0.1)
        elif act_fxn == "relu":
            self.act_fxn = nn.ReLU()
        elif act_fxn == "gelu":
            self.act_fxn = nn.GELU()

        # these implementation tricks below are from anthropic's
        # most recent paper
        Wd_initial = torch.randn(overcomplete_basis_size, embedding_size)
        norms = torch.sqrt(torch.sum(Wd_initial**2, dim=0))
        desired_norms = torch.rand(embedding_size) * 0.95 + 0.05
        scale_factors = desired_norms / norms
        self.decoder.weight = nn.Parameter(Wd_initial * scale_factors)

        self.encoder.weight = nn.Parameter(
            self.decoder.weight.detach().clone().t()
        )

    def forward(self, delta_z):
        delta_c = self.act_fxn(
            self.encoder(delta_z)
        )  # this is nonlin(M1.Tx + b)
        delta_z_hat = self.decoder(delta_c)  # this is M2c
        return delta_z_hat, delta_c


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


def orthogonality_penalty(weight):
    # Calculate WW^T
    identity = torch.eye(weight.size(0), device=weight.device)
    wwt = weight @ weight.t()
    # Penalize the deviation of WW^T from the identity matrix
    penalty = (wwt - identity).pow(2).sum()
    return penalty


def train(
    dataloader,
    sparse_dict_model,
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
        reconstruction_error = 0.0
        sparsity_penalty = 0.0
        for delta_z_list in dataloader:
            optimizer.zero_grad()
            with autocast():  # Enables mixed precision
                assert len(delta_z_list) == 1
                delta_z = delta_z_list[0]
                assert isinstance(delta_z, torch.Tensor)
                # todo: check why this appears as a list
                delta_z_hat, delta_c = sparse_dict_model(delta_z)
                reconstruction_error = loss_fxn(delta_z_hat, delta_z)

                sparsity_penalty = torch.sum(torch.norm(delta_c, p=1, dim=1))
                total_loss = (
                    reconstruction_error + float(args.alpha) * sparsity_penalty
                )
                alpha = alpha_scheduler.get_coeff(epoch)
                if args.model_type == "linv":
                    total_loss = orthogonality_penalty(
                        sparse_dict_model.encoder.weight
                    )
                else:
                    total_loss = (
                        reconstruction_error + alpha * sparsity_penalty
                    )

            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optim_scheduler.step(epoch)
            if args.model_type == "linv":
                with torch.no_grad():
                    # SVD based approach to enforce orthogonality
                    u, _, v = torch.svd(sparse_dict_model.encoder.weight)
                    sparse_dict_model.encoder.weight.data = u @ v.t()
                    sparse_dict_model.decoder.weight.data = (
                        sparse_dict_model.encoder.weight.t()
                    )
            if args.grad_clip_sae:
                torch.nn.utils.clip_grad_norm_(
                    parameters=sparse_dict_model.parameters(), max_norm=1
                )
            epoch_loss += total_loss.item()
        average_loss = epoch_loss / len(dataloader)
        losses.append(average_loss)
        if epoch % 100 == 0:
            print(f"Ending epoch {epoch}, Average Loss: {average_loss:.4f}\n")
            print(
                f"Loss breakup as {reconstruction_error:.4f} reconstruction, {sparsity_penalty:.4f} sparsity"
            )

    return losses


def main(args, device):
    x = None
    if args.data_type == "emb":
        embeddings_file = os.path.join(args.embedding_dir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_train = np.array(f["cfc1_train"]).squeeze()
            cfc2_train = np.array(f["cfc2_train"]).squeeze()

        x = cfc2_train - cfc1_train
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
        x = (x - mean) / (std + 1e-8)
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
        x_df_train = x_df.iloc[: int(config.split * config.size)]
        x_df_test = x_df.iloc[int(config.split * config.size) :]

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

        lin_ent_tf = generate_invertible_matrix(lin_ent_dim)
        x_ent = np.array([lin_ent_tf @ x_gt[i] for i in range(x_gt.shape[0])])
        if args.data_type == "gt_ent":
            x = x_ent

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
    if args.model_type == "la":
        sparse_dict_model = LinearAutoencoder(
            embedding_size=embedding_dim,
            overcomplete_basis_size=overcomplete_basis_size,
        )
    elif args.model_type == "linv":
        sparse_dict_model = LinearInvertible(
            embedding_size=embedding_dim,
            overcomplete_basis_size=overcomplete_basis_size,
        )
    elif args.model_type == "nla":
        sparse_dict_model = NonLinearAutoencoder(
            embedding_size=embedding_dim,
            overcomplete_basis_size=overcomplete_basis_size,
            act_fxn=args.act_fxn,
        )
    else:
        raise NotImplementedError
    sparse_dict_model.to(device)
    optimizer = optim.AdamW(
        sparse_dict_model.parameters(), lr=float(args.lr), weight_decay=1e-5
    )
    optim_scheduler = LinearDecayLR(optimizer, args.num_epochs)
    alpha_scheduler = AlphaScheduler(args.num_epochs, args.alpha)

    loss_fxn = torch.nn.MSELoss()
    losses = train(
        dataloader=loader,
        sparse_dict_model=sparse_dict_model,
        optimizer=optimizer,
        optim_scheduler=optim_scheduler,
        alpha_scheduler=alpha_scheduler,
        loss_fxn=loss_fxn,
        args=args,
    )
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(args.embedding_dir, timestamp_str + "_models")
    sparse_dict_model_dir = os.path.join(modeldir, "sparse_dict_model")
    if not os.path.exists(sparse_dict_model_dir):
        os.makedirs(sparse_dict_model_dir)
    sparse_dict_model_config_file = os.path.join(
        modeldir, "models_config.yaml"
    )
    sparse_dict_model_config = {
        "model_type": args.model_type,
        "act_fxn": args.act_fxn,
        "embedding_size": embedding_dim,
        "overcomplete_basis_size": overcomplete_basis_size,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "data_type": args.data_type,
        "num_epochs": args.num_epochs,
        "gradient_clipping_for_dict": args.grad_clip_sae,
    }
    with open(sparse_dict_model_config_file, "w") as file:
        yaml.dump(sparse_dict_model_config, file)
    sparse_dict_model_dict_path = os.path.join(
        sparse_dict_model_dir, "sparse_dict_model.pth"
    )
    torch.save(sparse_dict_model.state_dict(), sparse_dict_model_dict_path)
    if args.data_type == args.data_type == "gt":
        np.save(os.path.join(modeldir, "x_gt.npy"), x_gt)
    elif args.data_type == "gt_ent":
        np.save(os.path.join(modeldir, "x_gt.npy"), x_gt)
        np.save(os.path.join(modeldir, "x_ent.npy"), x_ent)
        np.save(os.path.join(modeldir, "lin_ent_tf.npy"), lin_ent_tf)
    # small test script here for now
    delta_z_test = np.asarray(
        (
            x_df_test[cfc_columns]
            .apply(lambda row: sum(row, []), axis=1)
            .tolist()
        )
    )  # the gt latents
    if args.data_type == "gt_ent":
        delta_z_test_ent = np.array(
            [
                lin_ent_tf @ delta_z_test[i]
                for i in range(delta_z_test.shape[0])
            ]
        )  # entangled latents
    else:
        raise ValueError
    sparse_dict_model_dict = torch.load(sparse_dict_model_dict_path)
    sparse_dict_model.load_state_dict(sparse_dict_model_dict)
    delta_z_test_ent = (
        torch.from_numpy(delta_z_test_ent).to(device).type(torch.float32)
    )
    delta_z_hat_test, delta_c_test = sparse_dict_model(delta_z_test_ent)
    delta_z_hat_test = delta_z_hat_test.detach().cpu().numpy()
    delta_c_test = delta_c_test.detach().cpu().numpy()
    disentanglement_scores = DisentanglementScores()

    mcc_score = disentanglement_scores.get_mcc_score(
        delta_z_hat_test,
        delta_z_test,
    )
    print(f"MCC on the test set is: {mcc_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument(
        "--data-type", default="emb", choices=["emb", "gt", "gt_ent"]
    )
    parser.add_argument(
        "--model-type", default="linv", choices=["linv", "la", "nla"]
    )
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--lr", type=float, default=float(5e-5))
    parser.add_argument("--alpha", type=float, default=float(0.001))
    parser.add_argument("--overcomplete-basis-factor", type=int, default=1)
    parser.add_argument(
        "--grad-clip-sae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the feature (default: False) by calling --grad-clip-sae and --no-grad-clip-sae for disabling",
    )
    parser.add_argument(
        "--act_fxn", default="leakyrelu", choices=["leakyrelu", "relu", "gelu"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
