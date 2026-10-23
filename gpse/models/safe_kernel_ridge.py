"""Segfault-safe KernelRidge variant.

Works around intermittent segfaults observed inside ``scipy.linalg.solve``
(``assume_a="pos"``) with the OpenBLAS build bundled in some scipy wheels:
the multithreaded LAPACK ``dposv`` path occasionally dereferences a null
pointer in its parallel kernels (``dtrsm_*copy``/``dgemv``), killing the
whole process. The crash is timing-sensitive and only ever manifests with
more than one BLAS thread.

``SafeKernelRidge`` keeps the expensive kernel-matrix computation fully
multithreaded but pins the tiny ``n_samples x n_samples`` Cholesky solve to
a single thread, which is both safe and effectively free at this size.
"""

from __future__ import annotations

from numbers import Real
import warnings

import numpy as np
from scipy import linalg
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits


class KernelNumericalError(ValueError):
    """A kernel candidate cannot be evaluated with the requested stable solver."""


class SafeKernelRidge(KernelRidge):
    """Single-threaded solves, with optional dimension scaling and strict numerics.

    ``gamma_scale`` is dimensionless: each fit uses gamma_scale / X.shape[1],
    after the pipeline has fitted its own feature selection and scaling.
    Explicit numeric ``gamma`` retains sklearn's absolute-value semantics.
    ``numerical_policy='raise'`` rejects failed/ill-conditioned solves instead
    of allowing sklearn's least-squares fallback. Legacy behavior is the API
    default; the packaged kernelridge_reg configuration opts into strict mode.
    """

    # Older joblib artifacts predate these constructor fields. Class defaults
    # keep their predictions and sklearn get_params/clone behavior compatible.
    gamma_scale = None
    numerical_policy = "legacy"

    def __init__(
        self,
        alpha=1,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        gamma_scale=None,
        numerical_policy="legacy",
    ):
        super().__init__(
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
        )
        self.gamma_scale = gamma_scale
        self.numerical_policy = numerical_policy

    def _get_kernel(self, X, Y=None):
        def compute():
            if self.gamma_scale is None:
                return super(SafeKernelRidge, self)._get_kernel(X, Y)
            from sklearn.metrics.pairwise import pairwise_kernels

            return pairwise_kernels(
                X,
                Y,
                metric=self.kernel,
                gamma=self.gamma_,
                degree=self.degree,
                coef0=self.coef0,
                filter_params=True,
            )

        if self.numerical_policy == "legacy":
            return compute()
        try:
            with np.errstate(over="raise", invalid="raise"):
                matrix = compute()
        except FloatingPointError as exc:
            raise KernelNumericalError(
                f"Non-finite {self.kernel} kernel computation: {exc}"
            ) from exc
        if not np.isfinite(matrix).all():
            raise KernelNumericalError(f"Non-finite {self.kernel} kernel matrix")
        return matrix

    def get_kernel_metadata(self):
        return {
            "kernel": self.kernel if isinstance(self.kernel, str) else "callable",
            "gamma": self.gamma,
            "gamma_scale": self.gamma_scale,
            "effective_gamma": getattr(self, "gamma_", None),
            "feature_count": getattr(self, "n_kernel_features_", None),
            "numerical_policy": self.numerical_policy,
        }

    def fit(self, X, y=None, sample_weight=None):
        if self.numerical_policy not in {"legacy", "raise"}:
            raise ValueError("numerical_policy must be 'legacy' or 'raise'")
        shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
        if len(shape) != 2 or shape[1] < 1:
            raise ValueError("KernelRidge requires a two-dimensional, nonempty feature matrix")
        self.n_kernel_features_ = int(shape[1])
        if self.gamma_scale is not None:
            if (
                not isinstance(self.gamma_scale, Real)
                or isinstance(self.gamma_scale, bool)
                or not np.isfinite(self.gamma_scale)
                or self.gamma_scale <= 0
            ):
                raise ValueError("gamma_scale must be a finite positive number")
            if self.gamma is not None:
                raise ValueError("Specify either absolute gamma or gamma_scale, not both")
            if self.kernel not in {"rbf", "laplacian", "polynomial", "poly", "sigmoid"}:
                raise ValueError(
                    "gamma_scale requires an RBF, laplacian, polynomial or sigmoid kernel"
                )
            self.gamma_ = float(self.gamma_scale) / self.n_kernel_features_
        else:
            self.gamma_ = self.gamma
            if (
                self.gamma is None
                and isinstance(self.kernel, str)
                and self.kernel
                in {
                    "rbf",
                    "laplacian",
                    "polynomial",
                    "poly",
                    "sigmoid",
                }
            ):
                self.gamma_ = 1.0 / self.n_kernel_features_
        original_solve = linalg.solve

        def single_threaded_solve(*args, **kwargs):
            with threadpool_limits(limits=1):
                if self.numerical_policy == "legacy":
                    return original_solve(*args, **kwargs)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", linalg.LinAlgWarning)
                        solution = original_solve(*args, **kwargs)
                except (np.linalg.LinAlgError, linalg.LinAlgWarning) as exc:
                    # A distinct ValueError bypasses sklearn's LinAlgError
                    # handler, so an invalid kernel never falls back to lstsq.
                    raise KernelNumericalError(
                        f"Unstable {self.kernel} kernel solve (alpha={self.alpha}, "
                        f"effective_gamma={self.gamma_}): {exc}"
                    ) from exc
                if not np.isfinite(solution).all():
                    raise KernelNumericalError(f"Non-finite {self.kernel} kernel solution")
                return solution

        linalg.solve = single_threaded_solve
        try:
            return super().fit(X, y=y, sample_weight=sample_weight)
        finally:
            linalg.solve = original_solve
