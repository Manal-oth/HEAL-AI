"""Institution-style non-IID partition following Eq. 35 and the 12-site layout in the article."""
import numpy as np

# 3 academic (35%), 5 community (45%), 2 specialty (12%), 2 rural (8%)
SITE_TYPES = [("academic", 3, 0.35), ("community", 5, 0.45), ("specialty", 2, 0.12), ("rural", 2, 0.08)]

def site_sizes(n_total: int):
    sizes = []
    for _, k, share in SITE_TYPES:
        sizes += [share / k] * k
    sizes = np.array(sizes); return sizes / sizes.sum()

def partition(y: np.ndarray, alpha: float, rng: np.random.RandomState, spread: float = 0.8):
    """P(D_k)=alpha*P_global+(1-alpha)*P_local(k): client positive-rate q_k mixes the global prevalence
    with a client-specific log-normal shift. Disjoint patient-level partition. Returns list of index arrays."""
    n = len(y); pi = y.mean(); w = site_sizes(n); K = len(w)
    local = pi * np.exp(rng.normal(0, spread, K))
    q = alpha * pi + (1 - alpha) * local
    q = np.clip(q * pi / (w * q).sum(), 1e-3, 0.9)
    pos = list(rng.permutation(np.where(y == 1)[0])); neg = list(rng.permutation(np.where(y == 0)[0]))
    out = []
    for k in range(K):
        m = int(round(w[k] * n)); npos = min(int(round(m * q[k])), len(pos)); nneg = min(m - npos, len(neg))
        idx = [pos.pop() for _ in range(npos)] + [neg.pop() for _ in range(nneg)]
        out.append(np.array(idx, dtype=int))
    return out
