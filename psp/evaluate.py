import numpy as np
import psp.metrics as metrics

from dataclasses import dataclass


@dataclass
class Evaluator:
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
                enc_gt @ self.delta_z_test[i] + self.b_e[i]
                for i in range(self.delta_z_test.shape[0])
            ]
        )
        mcc_gt_enc = metrics.mean_corr_coef(
            self.delta_c_test, delta_c_hat_enc_gt
        )
        import ipdb

        ipdb.set_trace()
        print(f"MCC b/w delta_c_enc_gt and delta_c: {mcc_gt_enc}")
