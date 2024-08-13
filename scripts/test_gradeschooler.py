import torch
from psp.sparse_dict import LinearAutoencoder


import argparse
import os
import yaml
from box import Box
import numpy as np
import pandas as pd


import psp.data_utils as data_utils
from psp.data_utils import tensorify, numpify

import psp.plot as psp_plot
from psp.evaluate import Evaluator


def prepare_test_data(args, device):
    data_config_file = os.path.join(args.embedding_dir, "config.yaml")
    with open(data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    model_1_config_file = os.path.join(args.model_dir_1, "models_config.yaml")
    model_2_config_file = os.path.join(args.model_dir_2, "models_config.yaml")
    with open(model_1_config_file, "r") as file:
        model_1_config = Box(yaml.safe_load(file))
    with open(model_2_config_file, "r") as file:
        model_2_config = Box(yaml.safe_load(file))
    model_1 = LinearAutoencoder(
        embedding_size=model_1_config.embedding_size,
        overcomplete_basis_size=model_1_config.overcomplete_basis_size,
    ).to(device)
    model_2 = LinearAutoencoder(
        embedding_size=model_1_config.embedding_size,
        overcomplete_basis_size=model_1_config.overcomplete_basis_size,
    ).to(device)
    model_1_file = os.path.join(
        args.model_dir_1, "sparse_dict_model", "sparse_dict_model.pth"
    )
    model_2_file = os.path.join(
        args.model_dir_2, "sparse_dict_model", "sparse_dict_model.pth"
    )
    model_1_dict = torch.load(model_1_file)
    model_2_dict = torch.load(model_2_file)
    model_1.load_state_dict(model_1_dict)
    model_2.load_state_dict(model_2_dict)
    seeds = []
    seeds.append(model_1_config.seed)
    seeds.append(model_2_config.seed)

    delta_z_test = None
    x_test, tf_ids, num_tfs = data_utils.load_test_data(args, data_config)
    delta_z_test = x_test
    delta_z_test = tensorify(delta_z_test, device)
    delta_z_hat_test_1, delta_c_hat_test_1 = model_1(delta_z_test)
    delta_z_hat_test_2, delta_c_hat_test_2 = model_1(delta_z_test)
    delta_c_hat_test_1 = numpify(delta_c_hat_test_1)
    delta_z_hat_test_1 = numpify(delta_z_hat_test_1)
    delta_c_hat_test_2 = numpify(delta_c_hat_test_2)
    delta_z_hat_test_2 = numpify(delta_z_hat_test_2)
    delta_z_test = [numpify(delta_z_test)]
    wd_1 = numpify(model_1.decoder.weight.data)
    wd_2 = numpify(model_2.decoder.weight.data)
    delta_z_hat_test = [delta_z_hat_test_1, delta_z_hat_test_2]
    delta_c_hat_test = [delta_c_hat_test_1, delta_c_hat_test_2]
    w_d = [wd_1, wd_2]
    b_e = [
        numpify(model_1_dict["bias_encoder"]),
        numpify(model_2_dict["bias_encoder"]),
    ]
    return (
        delta_z_test,
        delta_c_hat_test,
        delta_z_hat_test,
        w_d,
        b_e,
        num_tfs,
        tf_ids,
        seeds,
    )


def main(args, device):
    (
        delta_z_test,
        delta_c_hat_test,
        delta_z_hat_test,
        w_d,
        b_e,
        num_tfs,
        tf_ids,
        seeds,
    ) = prepare_test_data(args, device)

    evaluator = Evaluator(
        delta_z_test=delta_z_test,
        delta_c_hat_test=delta_c_hat_test,
        delta_z_hat_test=delta_z_hat_test,
        w_d=w_d,
        b_e=b_e,
        num_tfs=np.array(num_tfs),
        tf_ids=np.array(tf_ids),
        seeds=seeds,
    )
    evaluator.get_mcc()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("model_dir_1")
    parser.add_argument("model_dir_2")
    parser.add_argument("--perplexity", default=5.0)
    parser.add_argument("--sparsity-threshold", default=float(5e-2))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
