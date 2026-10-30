"""Uniform edge-level anomaly-detection metrics, used by every method so numbers are
directly comparable. All functions take (y_true, y_score) with y_score = P(malicious).
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1))
    thr = float(t[min(i, len(t) - 1)]) if len(t) else float("nan")
    return float(f1[i]), float(p[i]), float(r[i]), thr


def precision_at_k(y, s, k):
    k = min(int(k), len(y))
    if k <= 0:
        return float("nan")
    idx = np.argsort(-s)[:k]
    return float(np.asarray(y)[idx].mean())


def recall_at_k(y, s, k):
    k = min(int(k), len(y))
    if k <= 0 or y.sum() == 0:
        return float("nan")
    idx = np.argsort(-s)[:k]
    return float(np.asarray(y)[idx].sum() / y.sum())


def detection_latency(y, s, thr):
    """Time-ordered index of the first correctly flagged malicious edge (0-based).
    Lower = faster detection. -1 if none flagged."""
    y = np.asarray(y); s = np.asarray(s)
    for i in range(len(y)):
        if y[i] == 1 and s[i] >= thr:
            return int(i)
    return -1


def full_metrics(y, s, ks=(50, 100, 200), thr=None):
    y = np.asarray(y).astype(int); s = np.asarray(s).astype(float)
    out = {"n": int(len(y)), "n_pos": int(y.sum())}
    if y.max() == y.min():
        for k in ("auc", "ap"):
            out[k] = float("nan")
        out.update({"f1": float("nan"), "precision": float("nan"), "recall": float("nan")})
    else:
        out["auc"] = _safe(roc_auc_score, y, s)
        out["ap"] = _safe(average_precision_score, y, s)
        f1, p, r, t = best_f1(y, s)
        out.update({"f1": f1, "precision": p, "recall": r, "best_threshold": t})
        if thr is None:
            thr = t
    for k in ks:
        out[f"P@{k}"] = precision_at_k(y, s, k)
        out[f"R@{k}"] = recall_at_k(y, s, k)
    out["latency"] = detection_latency(y, s, thr if thr is not None else 0.5)
    return out


def fmt_table(rows, cols=("method", "dataset", "auc", "ap", "f1", "precision", "recall")):
    head = "  ".join(f"{c:>12s}" for c in cols)
    lines = [head]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else f"{str(v):>12s}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
