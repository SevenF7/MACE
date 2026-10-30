"""Dump per-edge test scores + labels for a trained LAMD/Jbeil checkpoint, so every
method can be scored with the SAME metric code (utils/edge_metrics.py).

Writes results/scores/{tag}.npz with: scores, labels, sources, timestamps.

Usage:
  python eval_lamd_edge_scores.py --data mas_ms_mmlu --prefix mt_lamd_mas_ms_mmlu \
      --induct 0 --val_time 3100002 --test_time 3900002 --edge_aware_decoder --use_memory
"""
import argparse, os, time, socket, json
import numpy as np
import torch

from model.tgn import TGN
from utils.utils import get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--induct", type=float, default=0.0)
    p.add_argument("--val_time", type=float, default=None)
    p.add_argument("--test_time", type=float, default=None)
    p.add_argument("--bs", type=int, default=200)
    p.add_argument("--n_degree", type=int, default=10)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=1)
    p.add_argument("--use_memory", action="store_true")
    p.add_argument("--edge_aware_decoder", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", default="results/scores")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    class L:
        def info(self, m):
            pass
    node_features, edge_features, full_data, train_data, val_data, test_data, _, _ = get_data(
        args.data, args.induct, 1000000, different_new_nodes_between_val_and_test=False,
        randomize_features=False, logger=L(), val_time=args.val_time, test_time=args.test_time)

    try:
        full_ngh = get_neighbor_finder(full_data, False, use_direction=True)
    except TypeError:  # Jbeil's finder has no use_direction kwarg
        full_ngh = get_neighbor_finder(full_data, False)
    ms, ss, md, sd = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    import inspect
    kwargs = dict(neighbor_finder=full_ngh, node_features=node_features, edge_features=edge_features,
                  device=device, n_layers=args.n_layer, n_heads=args.n_head, dropout=0.1,
                  use_memory=args.use_memory, message_dimension=100, memory_dimension=10,
                  memory_update_at_start=True, embedding_module_type="graph_attention",
                  message_function="identity", aggregator_type="last", memory_updater_type="gru",
                  n_neighbors=args.n_degree, mean_time_shift_src=ms, std_time_shift_src=ss,
                  mean_time_shift_dst=md, std_time_shift_dst=sd, one_hot_dir=True, dir_gnn=True,
                  use_transformer=True, edge_aware_decoder=args.edge_aware_decoder)
    accepted = set(inspect.signature(TGN.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    tgn = TGN(**kwargs).to(device)
    ckpt = args.ckpt or f"./saved_models/{args.prefix}-{args.data}.pth"
    tgn.load_state_dict(torch.load(ckpt, map_location=device))
    tgn.eval()

    sc, lb, src, ts, dst = [], [], [], [], []
    n = len(test_data.sources)
    with torch.no_grad():
        for k in range(0, n, args.bs):
            e = min(n, k + args.bs)
            sb, db = test_data.sources[k:e], test_data.destinations[k:e]
            tb, eb = test_data.timestamps[k:e], test_data.edge_idxs[k:e]
            pos, _ = tgn.compute_edge_probabilities(sb, db, db, tb, eb, args.n_degree)
            sc.append(pos.cpu().numpy().reshape(-1)); lb.append(np.asarray(test_data.labels[k:e]))
            src.append(np.asarray(sb)); ts.append(np.asarray(tb)); dst.append(np.asarray(db))
    tag = args.tag or args.prefix
    path = os.path.join(args.out_dir, f"{tag}.npz")
    np.savez(path, scores=np.concatenate(sc), labels=np.concatenate(lb).astype(int),
             sources=np.concatenate(src), destinations=np.concatenate(dst),
             timestamps=np.concatenate(ts))
    print(f"[eval_lamd_edge_scores] {tag}: n={len(np.concatenate(lb))} "
          f"pos={int(np.concatenate(lb).sum())} -> {path}")


if __name__ == "__main__":
    main()
