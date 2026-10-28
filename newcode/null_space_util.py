"""Null-subspace steering with row-vector hidden states.

For retain activations R in R^(n_r x d), Q contains selected right singular
vectors of R and P = Q Q^T.  Given forget activations H in R^(n_f x d) and
refusal direction r in R^d, fit

    min_D ||H P D - 1 r^T||_F^2 + lambda ||P D||_F^2,
    W = P D.

Inference applies h W (not W h) to a row-vector activation h.  A fixed
subspace fraction selects the smallest singular directions: unless their
singular values are zero, this is an *approximate* null space.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    if min(value.shape) == 0:
        raise ValueError(f"{name} must have nonempty sample and feature dimensions")
    if value.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value.to(dtype=torch.float64)


def _ratio(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _regularization(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("lambda_reg must be finite and nonnegative")


def null_space_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=None):
    """Return orthonormal Q of shape (d, k) for A of shape (n, d).

    With ``abs_nullspace_ratio=None``, use the numerical null space, enlarged
    to at least floor(d * min_null_space_ratio) directions.  An explicit
    absolute fraction fixes k=floor(d * fraction), including k=0.  Direct SVD
    avoids squaring the condition number via A.T @ A.
    """
    A = _matrix(A, "A")
    _ratio(min_null_space_ratio, "min_null_space_ratio")
    if abs_nullspace_ratio is not None:
        _ratio(abs_nullspace_ratio, "abs_nullspace_ratio")
    n_samples, n_features = A.shape
    _, singular_values, vh = torch.linalg.svd(A, full_matrices=n_samples < n_features)
    if abs_nullspace_ratio is None:
        tolerance = singular_values.max() * torch.finfo(A.dtype).eps * max(A.shape)
        rank = int((singular_values > tolerance).sum().item())
        count = max(n_features - rank, int(n_features * min_null_space_ratio))
    else:
        count = int(n_features * abs_nullspace_ratio)
    logger.info("Selected null-subspace dimension %d / %d", count, n_features)
    # -0 is 0 in Python; vh[-0:] would incorrectly select the entire space.
    return vh[n_features - count :].T.contiguous()


def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=None):
    """Return the d x d symmetric projector P=QQ^T."""
    Q = null_space_l(A, min_null_space_ratio, abs_nullspace_ratio)
    P = Q @ Q.T
    logger.debug("Retain projection residual: %g", torch.linalg.norm(A.double() @ P))
    return P


def cal_tilde_delta_with_regularization_l(
    H_h_layer, P_layer, refusal_vector, lambda_reg, device=None
):
    """Fit D (d x d) in the module-level regularized objective.

    ``device`` is retained for compatibility and, when specified, moves all
    inputs together.  With no device argument, computation follows H.
    """
    _regularization(lambda_reg)
    H = _matrix(H_h_layer, "H_h_layer")
    P = _matrix(P_layer, "P_layer").to(device=H.device)
    if P.shape != (H.shape[1], H.shape[1]):
        raise ValueError("P_layer must have shape (hidden_size, hidden_size)")
    r = torch.as_tensor(refusal_vector, dtype=H.dtype, device=H.device)
    if r.shape != (H.shape[1],) or not torch.isfinite(r).all():
        raise ValueError("refusal_vector must be a finite hidden_size vector")
    if device is not None:
        H, P, r = H.to(device), P.to(device), r.to(device)
    X = H @ P
    normal = X.T @ X + lambda_reg * (P.T @ P)
    # Every target row is r^T: solve one RHS instead of materializing n copies.
    coefficients = torch.linalg.pinv(normal, hermitian=True) @ X.sum(dim=0)
    return torch.outer(coefficients, r)


def cal_steering_matrix_l(P_layer, tilde_delta_layer, device=None):
    """Return W=P D on P's device unless an explicit device is requested."""
    P = _matrix(P_layer, "P_layer")
    D = _matrix(tilde_delta_layer, "tilde_delta_layer")
    if P.shape != D.shape or P.shape[0] != P.shape[1]:
        raise ValueError("P_layer and tilde_delta_layer must be equal square matrices")
    target = P.device if device is None else device
    return P.to(target) @ D.to(target)


def fit_steering_matrix(retain, forget, refusal_vector, nullspace_ratio, lambda_reg):
    """Fit W without a full d x d normal-equation pseudoinverse.

    Let Z=H Q.  For lambda>0, solve
        a=(Z^T Z + lambda I)^(-1) Z^T 1,  W=(Q a) r^T.
    For lambda=0 use a=pinv(Z)1, the minimum-norm least-squares solution.
    This is equivalent to the projected objective above because Q^T Q=I.
    """
    _regularization(lambda_reg)
    H = _matrix(forget, "forget")
    R = _matrix(retain, "retain").to(H.device)
    if H.shape[1] != R.shape[1]:
        raise ValueError("retain and forget hidden sizes must match")
    r = torch.as_tensor(refusal_vector, dtype=H.dtype, device=H.device)
    if r.shape != (H.shape[1],) or not torch.isfinite(r).all():
        raise ValueError("refusal_vector must be a finite hidden_size vector")
    Q = null_space_l(R, abs_nullspace_ratio=nullspace_ratio)
    if Q.shape[1] == 0:
        return H.new_zeros(H.shape[1], H.shape[1])
    Z = H @ Q
    if lambda_reg == 0:
        coefficients = torch.linalg.pinv(Z) @ H.new_ones(H.shape[0])
    else:
        normal = Z.T @ Z
        normal.diagonal().add_(lambda_reg)
        coefficients = torch.linalg.solve(normal, Z.sum(dim=0))
    W = torch.outer(Q @ coefficients, r)
    error = torch.linalg.norm(H @ W - r) / math.sqrt(H.shape[0])
    logger.info("Forget reconstruction RMS vector error: %g", error)
    return W
