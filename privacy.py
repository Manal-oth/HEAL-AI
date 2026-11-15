"""Client-level Gaussian mechanism with an RDP accountant (no subsampling amplification: conservative).

Adjacency: one client's whole update is added/removed. Per-round mechanism: sum of L2-clipped client updates
(clip C) + N(0, (z*C)^2 I). Gaussian RDP: eps_alpha = alpha / (2 z^2) per round; composes additively over T rounds.
"""
import math
import numpy as np

ORDERS = np.concatenate([np.linspace(1.05, 10, 60), np.arange(11, 257)])

def rdp_to_eps(z: float, T: int, delta: float) -> float:
    rdp = T * ORDERS / (2 * z * z)
    eps = rdp + math.log(1 / delta) / (ORDERS - 1)
    return float(eps.min())

def noise_multiplier(target_eps: float, T: int, delta: float) -> float:
    lo, hi = 1e-2, 1e5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if rdp_to_eps(mid, T, delta) > target_eps: lo = mid
        else: hi = mid
    return hi

def paper_eq34_total_eps(eps_round: float, T: int, delta: float) -> float:
    """Advanced composition as printed in the article's Eq. 34."""
    return math.sqrt(2 * T * math.log(1 / delta)) * eps_round + T * eps_round * (math.exp(eps_round) - 1)

def paper_eq29_sigma(C: float, eps: float, delta: float) -> float:
    """Classic single-release Gaussian calibration (article Eq. 29). Valid only for eps<=1."""
    return C * math.sqrt(2 * math.log(1.25 / delta)) / eps
