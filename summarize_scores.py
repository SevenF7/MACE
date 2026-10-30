"""Aggregate every method's dumped test scores into ONE comparable table.

Reads results/scores/{method}__{dataset}.npz (keys: scores, labels) and writes
  results/main_table.csv  +  docs/RESULTS_MAIN.md (markdown table)

Usage: python summarize_scores.py [--out docs/RESULTS_MAIN.md]
"""
import argparse, glob, json, os, re
import numpy as np

from utils.edge_metrics import full_metrics

METHOD_ORDER = ["LAMD", "Jebil", "Jbeil", "ProvCTDG", "StrGNN", "GeneralDyG", "SAD", "FFADE", "MIDAS"]


def load_all(score_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(score_dir, "*.npz"))):
        base = os.path.basename(f)[:-4]
        if "__" not in base:
            continue
        method, dataset = base.split("__", 1)
        d = np.load(f, allow_pickle=True)
        y = d["labels"]; s = d["scores"]
        m = full_metrics(y, s)
        m["method"] = method; m["dataset"] = dataset
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_dir", default="results/scores")
    ap.add_argument("--out", default="docs/RESULTS_MAIN.md")
    a = ap.parse_args()
    rows = load_all(a.score_dir)
    datasets = sorted(set(r["dataset"] for r in rows))
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    methods += sorted(set(r["method"] for r in rows) - set(methods))

    cols = ["auc", "ap", "f1", "precision", "recall", "P@100", "R@100", "latency", "n_pos"]
    os.makedirs("results", exist_ok=True)
    with open("results/main_table.csv", "w") as fh:
        fh.write("method,dataset," + ",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: (x["dataset"], x["method"])):
            fh.write(f"{r['method']},{r['dataset']}," +
                     ",".join(f"{r.get(c, float('nan')):.6f}" for c in cols) + "\n")

    lines = ["# 统一主表（Path A + chain 切分：train=chain0-3 / val=chain4 / test=chain5）", "",
             "所有方法共用同一份 `data/ml_{dataset}.csv` 与同一测试集；指标由 `utils/edge_metrics.py` 统一计算。", ""]
    hdr = "| dataset | method | AUC | AP | F1 | Prec | Rec | P@100 | R@100 | Latency | n_pos |"
    lines += [hdr, "|" + "---|" * 11]
    for ds in datasets:
        for m in methods:
            r = next((x for x in rows if x["dataset"] == ds and x["method"] == m), None)
            if r is None:
                continue
            lines.append(f"| {ds} | {m} | {r['auc']:.4f} | {r['ap']:.4f} | {r['f1']:.4f} | "
                         f"{r['precision']:.4f} | {r['recall']:.4f} | {r['P@100']:.4f} | "
                         f"{r['R@100']:.4f} | {r['latency']} | {r['n_pos']} |")
    lines += ["", "*（Latency = 按时间序首个被正确标记的恶意边的序号，越小越快；-1 表示全未命中）*"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[summarize_scores] {len(rows)} rows -> results/main_table.csv, {a.out}")


if __name__ == "__main__":
    main()
