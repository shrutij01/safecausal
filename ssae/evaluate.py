from re import L
import data_utils as utils
import metrics

from ssae import DictLinearAE

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from box import Box
from collections import Counter, defaultdict
import debug_tools as dbg

import itertools
import os
from nnsight import CONFIG, LanguageModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG.API.APIKEY = "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"


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def get_most_frequent_index(indices):
    if indices is None:
        print("The list is empty.")
        return None

    indices_list = [int(i.item()) for i in indices]
    counter = Counter(indices_list)
    most_frequent_index, count = counter.most_common(1)[0]
    print(
        f"The most common index is {most_frequent_index} with count {count} / {len(indices_list)}."
    )
    return most_frequent_index


def pca_transform(X: torch.Tensor):
    """
    Computes the first principal direction of X.

    Args:
        X (torch.Tensor): Input data of shape (n_samples, n_features)

    Returns:
        direction (torch.Tensor): First principal direction (unit vector of shape [n_features])
    """
    n, d = X.shape
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean
    U, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    U, Vh = svd_flip(U, Vh)
    components = Vh
    return torch.matmul(X - mean, components.t()), components, mean


def load_llamascope_checkpoint():
    model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"

    filename = "Llama3_1-8B-Base-L31R-8x/checkpoints/final.safetensors"

    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded checkpoint to: {filepath}")

    # 2. Load the safetensor
    state_dict = load_file(filepath)
    decoder_weight = state_dict[
        "decoder.weight"
    ]  # shape: (vocab_size, d_model)
    decoder_bias = state_dict["decoder.bias"]
    encoder_weight = state_dict["encoder.weight"]
    encoder_bias = state_dict["encoder.bias"]
    print(f"decoder.weight shape: {decoder_weight.shape}")

    return (decoder_weight, decoder_bias, encoder_weight, encoder_bias)


def get_concept_detection_logits(
    z, encoder_weight, encoder_bias, decoder_bias, args
):
    """
    Compute the logits for 1sp concept detection.

    Args:
        concept_projections: [B, C] tensor of concept projections

    Returns:
        logits: [B, C] tensor of logits
    """
    encoder_weight = encoder_weight.to(z.device)
    encoder_bias = encoder_bias.to(z.device)
    decoder_bias = decoder_bias.to(z.device)
    if args.modeltype == "llamascope":
        concept_projections = torch.nn.functional.relu(
            encoder_weight.to(torch.float32) @ z.T + encoder_bias.unsqueeze(1)
        ).T
    elif args.modeltype == "ssae":
        concept_projections = (
            encoder_weight.to(torch.float32)
            @ (z.T - decoder_bias.unsqueeze(1))  # enc_dim x batch_size
            + encoder_bias.unsqueeze(1)
        ).T  # batch_size x enc_dim
    else:
        raise ValueError
    concept_detection_activations = concept_projections.max(dim=1).values
    concept_detection_scores = (
        concept_detection_activations
        - concept_detection_activations.min(dim=0).values
    ) / (
        concept_detection_activations.max(dim=0).values
        - concept_detection_activations.min(dim=0).values
    )
    # concept_projections = (concept_projections > 0.1).float()
    eps = 1e-10
    concept_logits = torch.nn.functional.softmax(concept_projections, dim=1)

    # Element-wise multiplication of probabilities with their log
    entropy = -torch.sum(
        concept_logits * torch.log(concept_logits + eps), dim=1
    )
    mean_over_max_logits = concept_logits.max(dim=1).values.mean(dim=0)
    print("mean_over_max_logits", mean_over_max_logits)
    print("entropy", entropy.max(), entropy.min(), entropy.mean())
    return (mean_over_max_logits, entropy)


def load_model_config(modeldir: str) -> Box:
    config_path = os.path.join(modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config


def load_ssae(
    modeldir: str, dataconfig: Box
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a DictLinearAE model and return the decoder weights as a numpy array.
    """
    weight_path = os.path.join(modeldir, "weights.pth")

    # Load with detailed error handling
    try:
        state_dict = torch.load(weight_path, map_location="cpu")
        print("✅ torch.load() successful")
        print(f"Keys: {list(state_dict.keys())}")

        # Check each tensor
        for key, tensor in state_dict.items():
            print(f"{key}: {tensor.shape}, {tensor.dtype}")
            if torch.isnan(tensor).any():
                raise ValueError(f"❌ NaN detected in {key}")
            if torch.isinf(tensor).any():
                raise ValueError(f"❌ Inf detected in {key}")
            if tensor.numel() == 0:
                raise ValueError(f"❌ Empty tensor in {key}")
        return (
            state_dict["decoder.weight"].clone(),
            state_dict["decoder.bias"].clone(),
            state_dict["encoder.weight"].clone(),
            state_dict["encoder.bias"].clone(),
        )
    except Exception as e:
        raise ValueError(f"❌ Error during torch.load(): {e}")


def compute_all_pairwise_mccs(
    weight_matrices: list[torch.Tensor],
) -> list[float]:
    """
    Compute mean correlation coefficients (MCCs) between all model pairs.
    """
    mccs = []
    for i, j in itertools.combinations(range(len(weight_matrices)), 2):
        mcc = metrics.mean_corr_coef(
            weight_matrices[i],
            weight_matrices[j],
            method="pearson",
        )
        mccs.append(mcc)
    return mccs


def get_max_cos_and_steering_vector_for_concept(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
):
    """
    z: [B, D] tensor of original vectors
    decoder_weight: [V, D] decoder weight matrix (rows = token embeddings)

    Returns:
        mean and std of max cosine similarities over decoder directions
    """
    import ipdb

    ipdb.set_trace()
    z = F.normalize(z, dim=1)  # [B, D]
    z_tilde = F.normalize(z_tilde, dim=1)  # [B, D]
    # decoder = F.normalize(
    #     decoder_weight, dim=1
    # )  # [V, D] — columns as directions
    decoder = decoder_weight.to(z.device)
    decoder_bias = F.normalize(decoder_bias, dim=0)
    decoder_bias = decoder_bias.to(z.device)
    B, D = z.shape
    V = decoder.shape[0]

    # z: [B, D], decoder: [D, V]
    z_tilde_hat = (
        z.unsqueeze(2)
        + decoder.unsqueeze(0)
        + (decoder_bias.unsqueeze(0)).unsqueeze(2)
    )  # [B, D, V]
    z_tilde_hat = F.normalize(
        z_tilde_hat, dim=1
    )  # normalize shifted vectors: [B, D, V]

    z_tilde = z_tilde.unsqueeze(2)  # [B, D, 1]
    cosines = torch.bmm(z_tilde.transpose(1, 2), z_tilde_hat).squeeze(
        1
    )  # [B, V]
    max_cosines = cosines.max(dim=1).values  # [B]
    indices = cosines.argmax(dim=1)  # [B]
    most_frequent_index = get_most_frequent_index(indices=indices)
    steering_vector = (
        decoder[:, most_frequent_index]
        if most_frequent_index is not None
        else None
    )
    return max_cosines.mean().item(), max_cosines.std().item(), steering_vector


def get_ood_cosine_similarity(steering_vector, ood_data, decoder_bias):
    """
    Compute cosine similarity between steering vector and OOD data.
    """
    tilde_z_ood, z_ood = utils.load_test_data(
        datafile=ood_data,
    )
    tilde_z_ood = utils.tensorify(tilde_z_ood, device)
    tilde_z_ood = F.normalize(tilde_z_ood, dim=1)
    z_ood = utils.tensorify(z_ood, device)
    z_ood = F.normalize(z_ood, dim=1)
    decoder_bias = decoder_bias.to(z_ood.device)
    # steering vector is alreayd normalized
    tilde_hat_z_ood = (
        z_ood + steering_vector.unsqueeze(0) + decoder_bias.unsqueeze(0)
    )
    cosines = []
    tilde_hat_z_ood = utils.numpify(tilde_hat_z_ood)
    tilde_z_ood = utils.numpify(tilde_z_ood)
    for i in range(tilde_z_ood.shape[0]):
        cosines.append(
            cosine_similarity(
                tilde_z_ood[i].reshape(1, -1),
                tilde_hat_z_ood[i].reshape(1, -1),
            )
        )
    print("OOD Cosine Similarities:", np.mean(cosines), np.std(cosines))


def take_pca(z_test, tilde_z_test):
    shifts = utils.tensorify((tilde_z_test - z_test), device)

    shifts_transformed, components, mean = pca_transform(shifts.float())
    pca_vec = (
        (components.sum(dim=0, keepdim=True) + mean).mean(0)
        # .view(z_test[0].shape[0], z_test[0].shape[1])
    )
    z_test = utils.tensorify(z_test, device)
    z_pca = F.normalize(z_test) + pca_vec
    z_pca = F.normalize(z_pca)
    z_pca = utils.numpify(z_pca)
    cosines_pca = []
    for i in range(tilde_z_test.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z_test[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )
    return cosines_pca


def split_test_data_by_concept(tilde_z_test, z_test, concept_labels_test):
    concept_labels_test = np.array(concept_labels_test)

    # Get unique labels
    unique_labels = np.unique(concept_labels_test)

    # Initialize the dictionary
    split_tensors = {}
    # Efficiently gather indices for each label
    for label in unique_labels:
        mask = concept_labels_test == label
        indices = np.where(mask)[0]
        # Convert numpy indices to torch tensor on same device as input tensors
        indices_tensor = torch.tensor(
            indices, device=tilde_z_test.device, dtype=torch.long
        )
        split_tensors[label] = (
            tilde_z_test[indices_tensor],
            z_test[indices_tensor],
        )
    return split_tensors


def main(args):
    tilde_z_test, z_test = utils.load_test_data(
        datafile=args.datafile,
    )
    z_test = utils.tensorify(z_test, device)
    tilde_z_test = utils.tensorify(tilde_z_test, device)
    with open(args.dataconfig, "r") as file:
        dataconfig = Box(yaml.safe_load(file))
    concept_labels_test_path = os.path.dirname(os.path.abspath(args.datafile))
    concept_labels_test = utils.load_json(
        os.path.join(concept_labels_test_path, "concept_labels_test.json")
    )
    concept_test_sets = split_test_data_by_concept(
        tilde_z_test=tilde_z_test,
        z_test=z_test,
        concept_labels_test=concept_labels_test,
    )
    # contains keys for each concept, with values as tuples of tensors
    # (tilde_z_test, z_test)
    if args.modeltype == "llamascope":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = (
            load_llamascope_checkpoint()
        )
        decoder_weight_matrices = [decoder_weight]
        decoder_bias_vectors = [decoder_bias]
        encoder_weight_matrices = [encoder_weight]
        encoder_bias_vectors = [encoder_bias]
        concept_metrics = {}
        for concept, concept_test_set in concept_test_sets.items():
            tilde_z_test, z_test = concept_test_set
            # Compute cosine similarities for each concept
            mean_cos, std_cos, steering_vector = (
                get_max_cos_and_steering_vector_for_concept(
                    z_test,
                    tilde_z_test,
                    decoder_weight,
                    decoder_bias,
                )
            )
            print(f"Mean Cosine Similarity for concept {concept}: {mean_cos}")
            print(f"Std Cosine Similarity for concept {concept}:  {std_cos}")
            concept_metrics[concept] = {
                "mean_cos": mean_cos,
                "std_cos": std_cos,
            }
            if (
                steering_vector is not None
                and hasattr(args, "store_steering")
                and args.store_steering
            ):
                steering_dir_name = "steering_vector_" + str(args.modeltype)
                output_dir = (
                    f"{os.path.dirname(args.datafile)}/{steering_dir_name}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Save the steering vector
                torch.save(
                    steering_vector,
                    os.path.join(
                        output_dir,
                        f"steering_vector_concept_{concept}_llamascope.pt",
                    ),
                )

                print(
                    f"Saved steering vector for concept {concept} to {output_dir}"
                )

            if args.ood_data:
                dirname = os.path.dirname(os.path.abspath(args.ood_data))
                ood_concept_id = utils.load_json(
                    os.path.join(dirname, "concept_labels_test.json")
                )[0]
                if concept == ood_concept_id:
                    get_ood_cosine_similarity(
                        steering_vector, args.ood_data, decoder_bias
                    )
    elif args.modeltype == "ssae":
        # Load the decoder weights from the LinearSAE model
        # (assuming the model is already trained)
        modeldirs = args.modeldirs
        if len(modeldirs) < 2:
            raise ValueError(
                "You must provide at least two model directories."
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # the below weights and biases are on CPU
        with torch.no_grad():
            decoder_weight_matrices = [
                load_ssae(modeldir, dataconfig)[0] for modeldir in modeldirs
            ]
            decoder_bias_vectors = [
                load_ssae(modeldir, dataconfig)[1] for modeldir in modeldirs
            ]
            encoder_weight_matrices = [
                load_ssae(modeldir, dataconfig)[2] for modeldir in modeldirs
            ]
            encoder_bias_vectors = [
                load_ssae(modeldir, dataconfig)[3] for modeldir in modeldirs
            ]
        import ipdb

        ipdb.set_trace()
        print("Computing pairwise MCCs...")
        mccs = compute_all_pairwise_mccs(decoder_weight_matrices)

        mean_mcc = np.mean(mccs)
        std_mcc = np.std(mccs)

        print("\nPairwise MCCs:")
        for i, (a, b) in enumerate(
            itertools.combinations(range(len(modeldirs)), 2)
        ):
            print(f"Model {a+1} vs Model {b+1}: MCC = {mccs[i]:.4f}")
        print(f"\nMean MCC: {mean_mcc:.4f}")
        print(f"Std  MCC: {std_mcc:.4f}")
        import ipdb

        ipdb.set_trace()
        concept_metrics = {}
        for concept, concept_test_set in concept_test_sets.items():
            tilde_z_test, z_test = concept_test_set
            # Compute cosine similarities for each concept
            mean_cos, std_cos, steering_vector = (
                get_max_cos_and_steering_vector_for_concept(
                    z_test,
                    tilde_z_test,
                    decoder_weight_matrices[0],
                    decoder_bias_vectors[0],
                )
            )
            print(f"Mean Cosine Similarity for concept {concept}: {mean_cos}")
            print(f"Std Cosine Similarity for concept {concept}:  {std_cos}")
            concept_metrics[concept] = {
                "mean_cos": mean_cos,
                "std_cos": std_cos,
                "steering_vector": steering_vector,
            }
            if (
                steering_vector is not None
                and hasattr(args, "store_steering")
                and args.store_steering
            ):
                steering_dir_name = "steering_vector_" + str(args.modeltype)
                output_dir = (
                    f"{os.path.dirname(args.datafile)}/{steering_dir_name}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Save the steering vector
                torch.save(
                    steering_vector,
                    os.path.join(
                        output_dir,
                        f"steering_vector_concept_{concept}_llamascope.pt",
                    ),
                )

            if args.ood_data:
                dirname = os.path.dirname(os.path.abspath(args.ood_data))
                ood_concept_id = utils.load_json(
                    os.path.join(dirname, "concept_labels_test.json")
                )[0]
                if concept == ood_concept_id:
                    get_ood_cosine_similarity(
                        steering_vector, args.ood_data, decoder_bias_vectors[0]
                    )
    elif args.modeltype == "pca":
        cosines_pca = take_pca(z_test, tilde_z_test)
        print(
            "USING PCA cosine similarities",
            np.mean(cosines_pca),
            np.std(cosines_pca),
        )
    else:
        raise ValueError
    mean_over_max_logits, entropy = get_concept_detection_logits(
        z_test,
        encoder_weight_matrices[0],
        encoder_bias_vectors[0],
        decoder_bias_vectors[0],
        args,
    )
    print("Concept Detection Scores", mean_over_max_logits, entropy)
    print(f"Using {args.modeltype}, computing max cosine similarities for ")
    print("...", mean_cos, std_cos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile")
    parser.add_argument("dataconfig")
    parser.add_argument(
        "--modeltype", default="ssae", choices=["llamascope", "ssae"]
    )
    parser.add_argument(
        "--modeldirs",
        nargs="+",
        type=str,
        help="List of model directories to compare.",
    )
    parser.add_argument(
        "--ood-data", help="Path to OOD data file using the same dataconfig."
    )
    parser.add_argument(
        "--store-steering",
        action="store_true",
        help="Store steering vectors extracted from the model.",
    )
    parser.add_argument(
        "--evaluate-steering",
        action="store_true",
        help="Evaluate steering vectors on test prompts with Llama3 model",
    )
    parser.add_argument(
        "--steering-layer",
        "-sl",
        type=int,
        default=16,
        help="Starting layer index to apply steering intervention (applies to this layer and all following layers, default: 16)",
    )
    parser.add_argument(
        "--steering-alpha",
        "-sa",
        type=float,
        default=5.0,
        help="Steering strength multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--test-prompts",
        "-tp",
        type=str,
        default="default",
        choices=[
            "default",
            "sycophancy",
            "truthfulness",
            "refusal",
            "general",
        ],
        help="Select test prompt set based on concept type (default: default)",
    )
    args = parser.parse_args()
    with dbg.debug_on_exception():
        main(args)
