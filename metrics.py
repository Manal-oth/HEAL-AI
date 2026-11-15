import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def best_f1_threshold(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    f = [2 * ((p >= t) & (y == 1)).sum() / max(1, (p >= t).sum() + (y == 1).sum()) for t in ts]
    return float(ts[int(np.argmax(f))])

def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2,
                f1=2 * tp / max(1, 2 * tp + fp + fn), sensitivity=sens, specificity=spec)

def bootstrap_auroc_ci(y, p, n=500, seed=0):
    r = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        vals.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def loss_threshold_mia_auc(loss_members, loss_nonmembers):
    """Yeom-style loss-threshold membership inference; 0.5 = no leakage."""
    s = np.concatenate([-loss_members, -loss_nonmembers]); l = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    return float(roc_auc_score(l, s))
