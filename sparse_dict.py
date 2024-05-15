import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

import argparse

import utils


class TiedWeightAutoencoder(nn.Module):
    """x: differences between the embeddings of two contexts
    which differ in k-concepts
    M: rows of this are the sparse codes which express this diff
    c: ensures x is a sparse combination of M's rows
    """

    def __init__(self, input_size, hidden_size):
        super(TiedWeightAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)

    def forward(self, x):
        c = torch.relu(self.encoder(x))  # this is ReLU(Mx + b)
        x_hat = torch.matmul(c, self.encoder.weight.T)  # this is M.Tc
        return x_hat, c


def train(dataloader, model, optimizer, loss_fxn):
    scaler = GradScaler()
    losses = []
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for x in dataloader:
            optimizer.zero_grad()
            with autocast():  # Enables mixed precision
                x_hat, c = model(x)
                reconstruction_error = loss_fxn(x_hat, x)
                abs_loss = torch.abs(c).sum(dim=-1)
                l1_reg = abs_loss.sum() / x.shape[1]
                total_loss = reconstruction_error + l1_reg

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
    if args.data_type == "toy":
        x = utils.load_toy_dataset(args.task_type)
        input_dim = 2
        hidden_dim = 64
    else:
        raise NotImplementedError
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = TiedWeightAutoencoder(input_dim, hidden_dim)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = torch.nn.MSELoss()
    losses = train(
        dataloader=loader,
        model=model,
        optimizer=optimizer,
        loss_fxn=loss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_type", required=True, choices=["toy", "embedding"]
    )
    parser.add_argument("--num_epochs", default=500)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument(
        "--task-type", default="swap", choices=["swap", "cycle"]
    )
    parser.add_argument(
        "--dataset-length",
        default=10000,
    )
    parser.add_argument("--string-length", default=3)
    parser.add_argument("--cycle-length", default=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
