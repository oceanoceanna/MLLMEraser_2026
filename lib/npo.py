"""Causal sequence likelihoods and Negative Preference Optimization.

NPO reference implementation: https://github.com/licong-lin/negative-preference-optimization
Paper: https://arxiv.org/abs/2404.05868 (Eq. 3).
"""

import math

import torch
import torch.nn.functional as F


def sequence_nll(logits, labels):
    """Return summed target-token NLL per example, with the causal one-token shift."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            "Expected logits [batch, sequence, vocab] and matching labels [batch, sequence]"
        )
    if labels.shape[1] < 2 or labels.shape[0] == 0:
        raise ValueError("NLL requires a nonempty batch and at least two tokens")
    targets = labels[:, 1:].to(logits.device)
    if torch.any((targets != -100).sum(dim=1) == 0):
        raise ValueError("Every sequence must contain at least one unmasked target")
    # Compute only supervised positions, avoiding a full fp32 [B, T, V] copy
    # for image placeholders and prompt tokens.
    valid = targets != -100
    token_losses = F.cross_entropy(logits[:, :-1][valid].float(), targets[valid], reduction="none")
    batch_indices = valid.nonzero(as_tuple=True)[0]
    return token_losses.new_zeros(labels.shape[0]).scatter_add(0, batch_indices, token_losses)


def npo_loss(current_nll, reference_nll, beta):
    r"""L = -2/beta * mean_i log sigmoid(beta * (NLL_theta_i - NLL_ref_i)).

    Since NLL_theta - NLL_ref = -log(p_theta(y|x) / p_ref(y|x)),
    minimizing this loss decreases the forget-answer likelihood. Apply the
    nonlinear transform *before* averaging examples, never to a batch-mean CE.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if (
        current_nll.ndim != 1
        or current_nll.numel() == 0
        or current_nll.shape != reference_nll.shape
    ):
        raise ValueError("NPO requires equally sized nonempty vectors of per-example NLL")
    reference_nll = reference_nll.detach().to(current_nll.device)
    return -2 / beta * F.logsigmoid(beta * (current_nll - reference_nll)).mean()
