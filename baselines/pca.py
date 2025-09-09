import torch
from typing import Tuple, Optional


class PCA:
    """
    Principal Component Analysis implementation using PyTorch tensors.

    Supports both CPU and GPU tensors with proper device handling.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        device: Optional[torch.device] = None,
        keep_all_components: bool = False,
    ):
        """
        Initialize PCA.

        Args:
            n_components: Number of components to keep. If None and keep_all_components=False,
                        keeps all components. If keep_all_components=True, this parameter is ignored.
            device: Device to use for computations. If None, uses input tensor's device.
            keep_all_components: If True, keeps all components (original behavior).
                                If False, keeps only the first n_components (new behavior).
        """
        self.n_components = n_components
        self.device = device
        self.keep_all_components = keep_all_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self._fitted = False

    @staticmethod
    def svd_flip(
        u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sign correction to ensure deterministic output from SVD.

        Args:
            u: Left singular vectors
            v: Right singular vectors (transposed)

        Returns:
            Tuple of sign-corrected (u, v)
        """
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        i = torch.arange(u.shape[1], device=u.device)
        signs = torch.sign(u[max_abs_cols, i])
        u = u * signs
        v = v * signs.view(-1, 1)
        return u, v

    def fit(self, X: torch.Tensor) -> "PCA":
        """
        Fit PCA on the input data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            self
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {X.dim()}D")

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("Need at least 2 samples to compute PCA")

        # Use input tensor's device if not specified
        if self.device is None:
            self.device = X.device

        # Move to specified device if needed
        X = X.to(self.device)

        # Center the data
        self.mean_ = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean_

        # Perform SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # Apply sign correction for deterministic results
        U, Vh = self.svd_flip(U, Vh)

        # Store components (principal directions)
        if self.keep_all_components:
            # keep all components
            self.components_ = Vh
            explained_variance = (S**2) / (n_samples - 1)
        else:
            # keep top n_components by explained variance
            n_components = self.n_components or min(n_samples, n_features)
            self.components_ = Vh[:n_components]
            explained_variance = (S[:n_components] ** 2) / (n_samples - 1)

        # Compute explained variance ratio
        self.explained_variance_ratio_ = (
            explained_variance
            / (S**2).sum()
            * (n_samples - 1)  # Normalize by total variance
        )

        self._fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform data to principal component space.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fitted before transform")

        X = X.to(self.device)
        X_centered = X - self.mean_
        return torch.matmul(X_centered, self.components_.t())

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit PCA and transform data in one step.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: torch.Tensor) -> torch.Tensor:
        """
        Transform data back to original space.

        Args:
            X_transformed: Transformed data of shape (n_samples, n_components)

        Returns:
            Data in original space of shape (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fitted before inverse_transform")

        X_transformed = X_transformed.to(self.device)
        return torch.matmul(X_transformed, self.components_) + self.mean_

    def get_nth_component(self, n: int) -> torch.Tensor:
        """
        Get the nth principal component.

        Args:
            n: Index of the component to retrieve (0-based)

        Returns:
            nth principal component vector
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fitted first")

        return self.components_[n]

    def get_explained_variance_ratio(self) -> torch.Tensor:
        """
        Get explained variance ratio for each component.

        Returns:
            Explained variance ratio tensor
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fitted first")

        return self.explained_variance_ratio_


def pca_transform(
    X: torch.Tensor, n_components: Optional[int] = None, keep_all: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function for simple PCA transformation (backward compatibility).

    Args:
        X: Input data of shape (n_samples, n_features)
        n_components: Number of components to return. If None and keep_all=True,
                    returns all components (original behavior).
        keep_all: If True, keeps all components (original behavior).
                If False, keeps only first n_components.

    Returns:
        Tuple of (transformed_data, components, mean)
    """
    pca = PCA(n_components=n_components, keep_all_components=keep_all)
    transformed = pca.fit_transform(X)
    return transformed, pca.components_, pca.mean_
