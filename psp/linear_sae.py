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
from psp.data_utils import tensorify
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
            self.encoder_ln(self.encoder(delta_z)) + self.encoder_bias
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


class ConstrainedLinearSAE(cooper.ConstrainedMinimizationProblem):
    def __init__(self, model, sparse_level, batch_size, num_concepts):
        super().__init__(is_constrained=True)
        self.model = model
        self.criterion = nn.MSELoss(reduction="mean")
        self.sparse_level = sparse_level
        self.batch_size = batch_size
        self.num_concepts = num_concepts

    def compute_loss(self, delta_z, delt_z_hat):
        return self.criterion(delta_z, delt_z_hat)

    def closure(self, delta_z):
        """
        This closure function computes the model's loss and the constraints.
        :param inputs: Input tensor to the model
        :param targets: Target tensor for reconstruction comparison
        """
        delta_z_hat, concept_indicators = self.model(delta_z)

        self.loss = self.compute_loss(delta_z, delta_z_hat)
        # Compute the sparsity constraint as an inequality defect
        self.ineq_defect = (
            torch.sum(torch.abs(concept_indicators))
            / self.batch_size
            / self.num_concepts
            - self.sparse_level
        )

        return cooper.CMPState(
            loss=self.loss, ineq_defect=self.ineq_defect, eq_defect=None
        )


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


def load_training_data(
    args,
) -> tuple[DataLoader, DataLoader, np.ndarray, int, int]:
    config_file = os.path.join(args.datadir, "data_config.yaml")
    with open(config_file, "r") as file:
        config = Box(yaml.safe_load(file))
    delta_z_train = None
    delta_z_eval = None
    rep_dim = config.rep_dim
    num_concepts = config.num_concepts
    if config.dataset_name == "lang":
        embeddings_file = os.path.join(args.datadir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_train = np.array(f["cfc1_train"]).squeeze()
            cfc2_train = np.array(f["cfc2_train"]).squeeze()
            cfc1_eval = np.array(f["cfc1_eval"]).squeeze()
            cfc2_eval = np.array(f["cfc2_eval"]).squeeze()

        delta_z_train = tensorify(cfc2_train - cfc1_train, device)
        delta_z_eval = tensorify(cfc2_eval - cfc1_eval, device)

    elif config.dataset_name in ["synth1", "synth2", "synth3"]:
        df_file = os.path.join(
            args.datadir,
            str(config.dataset_name) + ".csv",
        )
        cfc_columns = config.cfc_column_names
        converters = {col: ast.literal_eval for col in cfc_columns}
        df = pd.read_csv(df_file, converters=converters)
        df_train = df.iloc[0 : int(config.train_split * config.size)]
        df_eval = df.iloc[
            int(config.train_split * config.size) : int(
                config.eval_split * config.size
            )
        ]

        def convert_to_list_of_ints(value):
            if isinstance(value, str):
                value = ast.literal_eval(value)
            return [int(x) for x in value]

        for column in df_train[cfc_columns]:
            df_train[column] = df_train[column].apply(convert_to_list_of_ints)

        delta_z_train = tensorify(
            np.asarray((df_train["Tx"].tolist())),
            device,
        )
        delta_z_eval = tensorify(
            np.asarray((df_eval["Tx"].tolist())),
            device,
        )
        sigma_c_train = tensorify(
            np.asarray((df_train["x"].tolist())),
            device,
        )
        sigma_c_eval = tensorify(
            np.asarray((df_eval["x"].tolist())),
            device,
        )
        delta_c_train = tensorify(
            np.asarray((df_train["delta_C"].tolist())),
            device,
        )
        delta_c_eval = tensorify(
            np.asarray((df_eval["delta_C"].tolist())),
            device,
        )
    else:
        raise NotImplementedError
    train_dataset = TensorDataset(
        delta_z_train,
        sigma_c_train,
        delta_c_train,
    )
    eval_dataset = TensorDataset(
        delta_z_eval,
        sigma_c_eval,
        delta_c_eval,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=delta_z_eval.shape[0],
        shuffle=True,
        num_workers=0,
    )
    with open(config.pickle_path, "rb") as file:
        global_C = pickle.load(file)
    return train_loader, eval_loader, global_C, rep_dim, num_concepts


def save(args, sae_model, config_dict, modeldir):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    modeldir = os.path.join(
        args.datadir,
        str(args.indicator_threshold) + str(args.seed),
        timestamp_str,
    )
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    config_file = os.path.join(modeldir, "model_config.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config_dict, file)
    sae_dict_path = os.path.join(sae_model, "sparse_dict_model.pth")
    torch.save(sae_model.state_dict(), sae_dict_path)


def train(
    train_loader,
    eval_loader,
    global_C,
    sae_model,
    cmp_model,
    coop_optimizer,
    formulation,
    logger,
    num_epochs,
    indicator_threshold,
):
    for epoch in range(int(num_epochs)):
        epoch_loss = 0.0
        sparsity_penalty_total = 0.0
        for delta_z, _, _ in train_loader:
            # this makes bn use batch statistics while training, doesn't have any
            # effect for gn or ln
            sae_model.train()
            coop_optimizer.zero_grad()
            lagrangian = formulation.composite_objective(
                cmp_model.closure, delta_z
            )
            formulation.custom_backward(lagrangian)
            coop_optimizer.step(cmp_model.closure, delta_z)

            # Normalize the decoder columns to unit length after the parameters update
            unit_norm_decoder_columns(cmp_model.model)

            # Adjust the gradients post-optimization step to maintain unit norms
            unit_norm_decoder_columns_grad_adjustment_(cmp_model.model)
            delta_z_hat, _ = sae_model(delta_z)
            import ipdb

            ipdb.set_trace()
            epoch_loss += cmp_model.compute_loss(delta_z_hat, delta_z)
            sparsity_penalty_total += cmp_model.ineq_defect.item()
        logger.logkv("total_loss", epoch_loss)
        logger.logkv("sparsity_penalty", sparsity_penalty_total)
        if epoch % 100 == 0:
            sae_model.eval()
            cmp_model.eval()
            with torch.no_grad():
                for delta_z, sigma_c, delta_c in eval_loader:
                    delta_z_hat, concept_indicators = sae_model(delta_z)
                    concept_indicator_ones_indices = (
                        concept_indicators > indicator_threshold
                    )
                    concept_indicator_ones = (
                        concept_indicators > torch.threshold
                    ).astype(int)
                    global_C_hat = sae_model.decoder.weight.data
                    sigma_c_hat = global_C_hat @ concept_indicator_ones
                    delta_c_hat = global_C_hat[
                        :, concept_indicator_ones_indices
                    ]
                    max_cols = max(delta_c_hat.shape[1], delta_c.shape[1])

                    def pad_matrix(matrix, target_cols):
                        pad_cols = target_cols - matrix.shape[1]
                        padded_matrix = np.pad(
                            matrix,
                            ((0, 0), (0, pad_cols)),
                            "constant",
                            constant_values=0,
                        )
                        return padded_matrix

                    if delta_c_hat.shape[1] < delta_c.shape[1]:
                        delta_c_hat = pad_matrix(delta_c_hat, max_cols)
                    elif delta_c.shape[1] < delta_c_hat.shape[1]:
                        delta_c = pad_matrix(delta_c, max_cols)

                    # todo eval of delta_c per sample
                    mcc_sigma_c = mean_corr_coef(sigma_c, sigma_c_hat)
                    mcc_delta_c = mean_corr_coef(delta_c, delta_c_hat)
                    mcc_global_C = mean_corr_coef(global_C, global_C_hat)
                    eval_loss = cmp_model.compute_loss(delta_z, delta_z_hat)
                    logger.logkv("mcc_sigma_c", mcc_sigma_c)
                    logger.logkv("mcc_delta_c", mcc_delta_c)
                    logger.logkv("mcc_global_C", mcc_global_C)
                    logger.logkv("eval_loss", eval_loss)
        logger.dumpkvs()


def main(args):
    config_dict = {
        "seed": args.seed,
        "dataset": args.datadir,
        "alpha": args.alpha,
        "primal_lr": args.primal_lr,
        "dual_lr": args.dual_lr,
        "norm_type": args.norm_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "indicator_threshold": args.indicator_threshold,
    }
    set_seeds(int(args.seed))
    logger = Logger(project="psp", config=config_dict)
    train_loader, eval_loader, global_C, rep_dim, num_concepts = (
        load_training_data(args)
    )
    # Assuming the LinearSAE model and other parameters are already defined:
    sae_model = LinearSAE(
        rep_dim=rep_dim, num_concepts=num_concepts, norm_type=args.norm_type
    )
    cmp_model = ConstrainedLinearSAE(
        model=sae_model,
        sparse_level=float(args.alpha),
        batch_size=int(args.batch_size),
        num_concepts=num_concepts,
    )

    # Define optimizers
    primal_optimizer = cooper.optim.ExtraAdam(
        list(sae_model.parameters()), lr=args.primal_lr
    )
    dual_optimizer = cooper.optim.partial_optimizer(
        cooper.optim.ExtraAdam, lr=args.dual_lr
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
        eval_loader=eval_loader,
        global_C=global_C,
        sae_model=sae_model,
        cmp_model=cmp_model,
        coop_optimizer=coop_optimizer,
        formulation=formulation,
        logger=logger,
        num_epochs=args.num_epochs,
        indicator_threshold=args.indicator_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1100)
    parser.add_argument("--primal-lr", type=float, default=0.0001)
    parser.add_argument("--dual-lr", type=float, default=0.00005)
    parser.add_argument("--alpha", type=float, default=float(0.0001))
    parser.add_argument("--indicator-threshold", type=float, default=0.01)
    parser.add_argument("--num-concepts", type=int, default=3)
    parser.add_argument(
        "--norm-type", default="bn", choices=["ln", "gn", "bn"]
    )
    parser.add_argument("--seed", default=0)

    args, unknown = parser.parse_known_args()

    main(args)
