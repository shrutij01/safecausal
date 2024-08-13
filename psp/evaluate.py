import numpy as np
import psp.metrics as metrics
from sklearn.linear_model import LinearRegression

from dataclasses import dataclass
import psp.data_utils as data_utils


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
                self.delta_c_test, self.delta_c_hat_test
            )
            print(f"MCC b/w delta_c and delta_c_hat: {mcc_latents}")
            r2_latents = metrics.r2(self.delta_c_test, self.delta_c_hat_test)
            print(f"R2 b/w delta_c and delta_c_hat: {r2_latents}")

        if self.w_d_gt is not None:
            mcc_encoding = metrics.mean_corr_coef(self.w_d, self.w_d_gt)
            print(f"MCC b/w gt_encoding and learnt_encoding: {mcc_encoding}")
            r2_encoding = metrics.r2(self.delta_c_test, self.delta_c_hat_test)
            print(f"R2 b/w gt_encoding and learnt_encoding: {r2_encoding}")

    def get_metric_bounds(self):
        enc_gt = np.linalg.inv(self.w_d_gt)
        delta_c_hat_enc_gt = np.array(
            [
                enc_gt @ self.delta_z_test[i] + self.b_e
                for i in range(self.delta_z_test.shape[0])
            ]
        )
        mcc_gt_enc = metrics.mean_corr_coef(
            self.delta_c_test, delta_c_hat_enc_gt
        )
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
class Evaluator:
    # data to be a list with seeds as indices
    delta_z_test: list
    delta_c_hat_test: list
    delta_z_hat_test: list
    w_d: list
    b_e: list
    num_tfs: np.ndarray
    tf_ids: np.ndarray
    seeds: list

    def get_mcc(self):
        z_1, z_2 = data_utils.get_rep_pairs(
            50, self.delta_c_hat_test[0], self.delta_c_hat_test[1]
        )
        mcc_latents = metrics.mean_corr_coef(z_1, z_2)
        print(
            f"MCC between delta_c from a couple different runs {mcc_latents}..."
        )
        reg = LinearRegression().fit(z_1, z_2)
        score = reg.score(z_1, z_2)
        print(f"... and its OLS score {score}")
        mcc_encoding = metrics.mean_corr_coef(self.w_d[0], self.w_d[1])
        print(
            f"MCC between delta_c from a couple different runs {mcc_encoding}..."
        )
        reg = LinearRegression().fit(self.w_d[0], self.w_d[1])
        score = reg.score(self.w_d[0], self.w_d[1])
        print(f"... and its OLS score {score}")

    def compare_with_md(self):
        import ipdb

        ipdb.set_trace()
