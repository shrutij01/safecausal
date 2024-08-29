import numpy as np
import psp.metrics as metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity


from dataclasses import dataclass
import psp.data_utils as data_utils
import seaborn as sns
import matplotlib.pyplot as plt

import itertools
import statistics


@dataclass
class EvaluatorGT:
    delta_c_test: np.ndarray
    delta_z_test: np.ndarray
    delta_c_hat_test: np.ndarray
    delta_z_hat_test: np.ndarray
    w_d_gt: np.ndarray
    w_d: np.ndarray
    b_e: np.ndarray
    num_tfs: np.ndarray
    tf_ids: np.ndarray

    def run_evaluations(self, plot=False):
        if self.delta_c_test is not None:
            mcc_latents = metrics.mean_corr_coef(
                self.delta_c_test, self.delta_c_hat_test, method="pearson"
            )
            print(f"MCC b/w delta_c and delta_c_hat: {mcc_latents}")

        if self.w_d_gt is not None:
            mcc_encoding = metrics.mean_corr_coef(
                self.w_d, self.w_d_gt, method="pearson"
            )
            print(f"MCC b/w gt_encoding and learnt_encoding: {mcc_encoding}")

    def get_metric_bounds(self):
        enc_gt = np.linalg.inv(self.w_d_gt)
        delta_c_hat_enc_gt = np.array(
            [
                enc_gt @ self.delta_z_test[i]
                for i in range(self.delta_z_test.shape[0])
            ]
        )
        delta_c_hat_enc_gt_2 = np.array(
            [
                enc_gt @ self.delta_z_test[i] + self.b_e
                for i in range(self.delta_z_test.shape[0])
            ]
        )
        mcc_gt_enc = metrics.mean_corr_coef(
            self.delta_c_test, delta_c_hat_enc_gt, method="pearson"
        )
        import ipdb

        ipdb.set_trace()
        # metrics.mean_corr_coef(self.delta_c_test, self.delta_c_test) # 1.0000000000000002
        # metrics.mean_corr_coef(delta_c_hat_enc_gt, delta_c_hat_enc_gt) # 1.0000000000000002
        # self.delta_c_test.all() == delta_c_hat_enc_gt.all() # True
        # metrics.mean_corr_coef(self.delta_c_test, delta_c_hat_enc_gt) # 0.937802229649176
        # metrics.mean_corr_coef(self.delta_c_test, delta_c_hat_enc_gt) # 0.9999999999999999
        # delta_c_hat_enc_gt_2.all() == self.delta_c_test.all() # False
        # metrics.mean_corr_coef(self.delta_c_test, delta_c_hat_enc_gt_2, method="pearson") # with the bias term # 1.0
        # metrics.mean_corr_coef(self.delta_c_test, delta_c_hat_enc_gt, method="spearman") # 0.7557189483824308
        reg_1 = LinearRegression().fit(delta_c_hat_enc_gt, self.delta_c_test)
        reg_2 = LinearRegression().fit(
            self.delta_c_hat_test, self.delta_c_test
        )
        score_1 = reg_1.score(delta_c_hat_enc_gt, self.delta_c_test)
        score_2 = reg_2.score(self.delta_c_hat_test, self.delta_c_test)
        import ipdb

        ipdb.set_trace()
        print(f"MCC b/w delta_c_enc_gt and delta_c: {mcc_gt_enc}")
        print(f"OLS score b/w delta_c_hat and delta_c: {score_2}")
        print(
            f"OLS score b/w delta_c_enc_gt and delta_c: {score_1}, {score_2}"
        )


@dataclass
class EvaluatorSeeds:
    # data to be a list with seeds as indices
    delta_z_test: list
    delta_c_hat_test: list
    delta_z_hat_test: list
    w_d: list
    num_tfs: np.ndarray
    tf_ids: np.ndarray
    seeds: list
    z_test: np.ndarray
    z_tilde_test: np.ndarray

    def compute_pairwise_mcc(self):
        pairs = list(itertools.combinations(self.seeds, 2))
        mccs = []
        for pair in pairs:
            mccs.append(
                metrics.mean_corr_coef(
                    self.w_d[int(pair[0])],
                    self.w_d[int(pair[1])],
                    method="pearson",
                ),
            )
        return mccs

    def get_mcc_udr(self):
        mccs = self.compute_pairwise_mcc()
        print(f"Max MCC [w_d_i, w_d_j] {max(mccs)}")
        reg = LinearRegression().fit(self.w_d[0], self.w_d[1])
        score = reg.fit(self.w_d[0], self.w_d[1])
        print(f"ols score {score}")


@dataclass
class EvaluatorMD:
    md_1: np.ndarray
    md_2: np.ndarray
    encoded_md_1: np.ndarray
    encoded_md_2: np.ndarray
    delta_c_hat_1: np.ndarray
    delta_c_hat_2: np.ndarray

    def get_hotnesses(self, threshold=0.01):
        def l0(arr, threshold):
            # arr of batch_size x features
            l0_batch = np.array(
                [
                    arr[i][arr[i] >= threshold].shape[0]
                    for i in range(arr.shape[0])
                ]
            )
            return np.mean(l0_batch)

        import ipdb

        ipdb.set_trace()
        print(
            f"hotness of MD(orange --> purple) {l0(self.encoded_md_1, threshold)}"
        )
        print(
            f"hotness of MD(pruple --> pink) {l0(self.encoded_md_2, threshold)}"
        )
        print(
            f"by contrast, hotness of delta_c_hat(orange --> purple) {l0(self.delta_c_hat_1, threshold)}"
        )
        print(
            f"by contrast, hotness of delta_c_hat(purple --> pink) {l0(self.delta_c_hat_2, threshold)}"
        )
