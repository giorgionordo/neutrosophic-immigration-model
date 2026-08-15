# ============================================================================
# Project: A Neutrosophic Agent-Based Network Model for Immigration and Coexistence
# Main author: Giorgio Nordo
# Affiliation: Department of Mathematical and Computer Sciences, Physical Sciences
#              and Earth Sciences (MIFT), University of Messina, Italy
# E-mail: giorgio.nordo@unime.it
# Website: https://www.nordo.it
# Coauthors of the related paper: Carmelo Filippo Munafò, Nivetha Martin
# Suggested repository: https://github.com/giorgionordo/neutrosophic-immigration-model
# ============================================================================
"""SVNS primitives used by the agent-based immigration model."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

Triplet = Tuple[float, float, float]


def clamp01(value: float) -> float:
    """Project a scalar value onto the interval [0, 1]."""
    return float(min(1.0, max(0.0, value)))


def project_triplet(values: Iterable[float]) -> Triplet:
    """Componentwise projection of a triple onto [0, 1]^3."""
    values = tuple(values)
    if len(values) != 3:
        raise ValueError("A single-valued neutrosophic attitude must have three components.")
    return tuple(clamp01(float(v)) for v in values)  # type: ignore[return-value]


def score_lambda(attitude: Triplet, lam: float = 0.6) -> float:
    """Return Sc_lambda(<T, I, F>) = T - lambda F - (1-lambda) I."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must belong to [0, 1].")
    T, I, F = attitude
    return float(T - lam * F - (1.0 - lam) * I)


def score_distance(a: Triplet, b: Triplet, lam: float = 0.6) -> float:
    """Absolute distance between scalar neutrosophic scores."""
    return abs(score_lambda(a, lam) - score_lambda(b, lam))


def vector_distance(a: Triplet, b: Triplet) -> float:
    """Euclidean distance between two SVNS triples."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def gaussian_similarity(a: Triplet, b: Triplet, sigma: float = 0.55, lam: float = 0.6) -> float:
    """Gaussian compatibility based on the score distance."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    d = score_distance(a, b, lam)
    return float(math.exp(-(d * d) / (2.0 * sigma * sigma)))


def positive_activation(u: float) -> float:
    """Bounded positive-utility activation phi(u)=max(0,u)/(1+|u|)."""
    return float(max(0.0, u) / (1.0 + abs(u)))
