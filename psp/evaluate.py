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
class Evaluator:
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
            import ipdb

            ipdb.set_trace()
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
        udr = statistics.median(mccs)
        print(f"UDR Score {udr}")

    def ac_compare_with_md(self):
        ac_tf_ids = self.tf_ids == 12
        md_ac = np.mean(self.delta_z_test[0][ac_tf_ids], axis=0)
        delta_c_ac = np.mean(self.delta_c_hat_test[0][ac_tf_ids], axis=0)
        z_ac = self.z_test[ac_tf_ids]
        z_tilde_ac = self.z_tilde_test[ac_tf_ids]
        z_tilde_md = z_ac + md_ac
        z_tilde_delta_c = z_ac + delta_c_ac

        def get_cosine_similarities(embeddings_1, embeddings_2):
            assert embeddings_1.shape[0] == embeddings_2.shape[0]
            random_indices = np.random.choice(
                embeddings_1.shape[0], 20, replace=False
            )
            similarities = cosine_similarity(
                embeddings_1[random_indices], embeddings_2[random_indices]
            )[0]
            return similarities

        sim_md = get_cosine_similarities(z_tilde_md, z_tilde_ac)
        sim_delta_c = get_cosine_similarities(z_tilde_delta_c, z_tilde_ac)
        sns.kdeplot(sim_md, fill=True, color="blue", label="z + md")
        sns.kdeplot(sim_delta_c, fill=True, color="green", label="z + delta_c")
        plt.title("KDE of Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("kde_ac_nosp.png")
        plt.close()

    def objects_compare_with_md(self):
        object_tf_ids = self.tf_ids == 3
        md_objects = np.mean(self.delta_z_test[0][object_tf_ids], axis=0)
        delta_c_objects = np.mean(
            self.delta_c_hat_test[0][object_tf_ids], axis=0
        )
        z_objects = self.z_test[object_tf_ids]
        z_tilde_objects = self.z_tilde_test[object_tf_ids]
        z_tilde_md = z_objects + md_objects
        z_tilde_delta_c = z_objects + delta_c_objects

        def get_cosine_similarities(embeddings_1, embeddings_2):
            assert embeddings_1.shape[0] == embeddings_2.shape[0]
            # random_indices = np.random.choice(
            #     embeddings_1.shape[0], 700, replace=False
            # )
            similarities = cosine_similarity(embeddings_1, embeddings_2)[0]
            return similarities

        sim_md = get_cosine_similarities(z_tilde_md, z_tilde_objects)
        sim_delta_c = get_cosine_similarities(z_tilde_delta_c, z_tilde_objects)
        sns.kdeplot(sim_md, fill=True, color="blue", label="z + md")
        sns.kdeplot(sim_delta_c, fill=True, color="green", label="z + delta_c")
        plt.title("KDE of Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("kde_objects_nosp.png")
        plt.close()
