"""NumPy MLP + FedAvg / FedProx / FedNova-style / DP-FedAvg / centralized training."""
from __future__ import annotations
import numpy as np
from .privacy import noise_multiplier

class MLP:
    def __init__(self, d, h=64):
        self.d, self.h = d, h; self.sizes = [d * h, h, h, 1]; self.n = sum(self.sizes)
    def init(self, rng):
        W1 = rng.randn(self.d, self.h) * np.sqrt(2 / self.d); W2 = rng.randn(self.h) * np.sqrt(1 / self.h)
        return np.concatenate([W1.ravel(), np.zeros(self.h), W2, [0.0]])
    def unpack(self, th):
        a, b, c = self.d * self.h, self.d * self.h + self.h, self.d * self.h + 2 * self.h
        return th[:a].reshape(self.d, self.h), th[a:b], th[b:c], th[c]
    def predict(self, th, X):
        W1, b1, W2, b2 = self.unpack(th); h = np.maximum(X @ W1 + b1, 0)
        return 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -30, 30)))
    def loss_grad(self, th, X, y, pw):
        W1, b1, W2, b2 = self.unpack(th); a = X @ W1 + b1; h = np.maximum(a, 0)
        p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -30, 30))); wt = np.where(y == 1, pw, 1.0)
        dz = wt * (p - y) / wt.sum()
        gW2 = h.T @ dz; gb2 = dz.sum(); dh = np.outer(dz, W2) * (a > 0)
        return np.concatenate([(X.T @ dh).ravel(), dh.sum(0), gW2, [gb2]])
    def per_sample_loss(self, th, X, y):
        p = np.clip(self.predict(th, X), 1e-7, 1 - 1e-7); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def local_train(model, th0, X, y, rng, lr, epochs, bs, pw, mu=0.0, thg=None):
    th = th0.copy(); m = np.zeros_like(th); v = np.zeros_like(th); t = 0; n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            i = order[s:s + bs]
            if len(i) < 2: continue
            g = model.loss_grad(th, X[i], y[i], pw)
            if mu > 0: g = g + mu * (th - thg)
            t += 1; m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            th = th - lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return th, t

def run_federated(method, model, clients, Xtr, ytr, rng, rounds=150, frac=(0.3, 0.5), lr=3e-4, epochs=1, bs=64,
                  pw=3.0, mu=0.0, eps=None, delta=1e-5, clip=1.0):
    """method in {fedavg, fedprox, fednova, dpfedavg}. Returns final global params and a log dict."""
    th = model.init(rng); K = len(clients); log = {"rounds": rounds}
    z = noise_multiplier(eps, rounds, delta) if method == "dpfedavg" else 0.0
    if method == "dpfedavg": log.update(noise_multiplier=z, eps_target=eps, delta=delta, clip=clip)
    for r in range(rounds):
        m = max(2, int(round(rng.uniform(*frac) * K))); sel = rng.choice(K, m, replace=False)
        deltas, ns, taus = [], [], []
        for k in sel:
            idx = clients[k]
            if len(idx) < 2: continue
            thk, tau = local_train(model, th, Xtr[idx], ytr[idx], rng, lr, epochs, bs, pw,
                                   mu if method == "fedprox" else 0.0, th)
            deltas.append(thk - th); ns.append(len(idx)); taus.append(max(tau, 1))
        ns = np.array(ns, float); taus = np.array(taus, float); D = np.array(deltas)
        if method == "dpfedavg_paper":  # article Eq.28-29/33: each client clips, adds N(0,(sigma*C)^2) to its own update, weighted avg
            from .privacy import paper_eq29_sigma
            nrm = np.linalg.norm(D, axis=1, keepdims=True); D = D * np.minimum(1, clip / (nrm + 1e-12))
            D = D + rng.normal(0, paper_eq29_sigma(clip, eps, delta), D.shape); p = ns / ns.sum(); th = th + (p[:, None] * D).sum(0)
        elif method == "dpfedavg":
            nrm = np.linalg.norm(D, axis=1, keepdims=True); D = D * np.minimum(1, clip / (nrm + 1e-12))
            th = th + (D.sum(0) + rng.normal(0, z * clip, th.shape)) / len(D)
        elif method == "fednova":
            p = ns / ns.sum(); d = D / taus[:, None]; th = th + (p * taus).sum() * (p[:, None] * d).sum(0)
        else:
            p = ns / ns.sum(); th = th + (p[:, None] * D).sum(0)
    return th, log

def run_centralized(model, Xtr, ytr, rng, epochs=30, lr=3e-4, bs=64, pw=3.0):
    th, _ = local_train(model, model.init(rng), Xtr, ytr, rng, lr, epochs, bs, pw)
    return th, {"epochs": epochs}
