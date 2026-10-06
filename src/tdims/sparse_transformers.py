# Sparse sklearn helpers used with TDiMS matrices.
# Adapted from IBM/materials models/tdims (Apache-2.0).

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel


class SparseFeatureFilterVCSR(BaseEstimator, TransformerMixin):
    def __init__(self, min_samples: int = 30, verbose: bool = True):
        self.min_samples = min_samples
        self.verbose = verbose

    def fit(self, X, y=None):
        if issparse(X):
            feature_counts = np.asarray(X.getnnz(axis=0)).ravel()
        else:
            feature_counts = np.sum(X != 0, axis=0)

        self.support_ = feature_counts >= self.min_samples
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = int(np.sum(self.support_))

        if self.verbose:
            print(
                f"SparseFeatureFilter: {self.n_features_in_} → {self.n_features_out_} features"
            )
        return self

    def transform(self, X):
        if not hasattr(self, "support_"):
            raise AttributeError(
                "SparseFeatureFilterVCSR is not fitted yet. Call fit() before transform()."
            )
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        if not hasattr(self, "support_"):
            raise AttributeError(
                "SparseFeatureFilterVCSR is not fitted yet. Call fit() first."
            )
        return np.where(self.support_)[0] if indices else self.support_


class ToCSR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if issparse(X):
            return X.tocsr()
        return csr_matrix(X)


class ClipGreaterThanOneToZero(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if issparse(X):
            X = X.tocsr(copy=True)
            if X.nnz == 0:
                return X
            mask = X.data > self.threshold
            if np.any(mask):
                X.data[mask] = 0.0
                X.eliminate_zeros()
            return X
        X = np.array(X, copy=True)
        X[X > self.threshold] = 0.0
        return csr_matrix(X)


class SafeSelectFromModel(SelectFromModel):
    """If no feature is selected, keep at least one (top-1 by score)."""

    def _get_support_mask(self):
        mask = super()._get_support_mask()
        if mask is None:
            return mask
        if np.sum(mask) > 0:
            return mask

        est = self.estimator_
        if hasattr(est, "coef_"):
            scores = np.abs(np.ravel(est.coef_))
        elif hasattr(est, "feature_importances_"):
            scores = np.ravel(est.feature_importances_)
        else:
            mask[0] = True
            return mask

        if scores.size == 0:
            return mask

        best = int(np.nanargmax(scores))
        mask[best] = True
        return mask
