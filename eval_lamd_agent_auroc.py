"""Aggregate LAMD edge-level anomaly scores to agent level, for a node-level AUROC
comparable with the MAS GAD baselines (G-Safeguard / BlindGuard / DOMINANT / TAM / PREM).

Protocol
--------
- LAMD is the supervised edge classifier: p(edge) = P(malicious); anomaly score = p(edge).
- Score every TEST edge (time-ordered streaming with memory, same as training-time eval).
- Aggregate per SOURCE agent (the agent that emitted the message):
    * 'out_max'  : max anomaly score over the agent's outgoing edges
    * 'out_mean' : mean anomaly score over the agent's outgoing edges
  (and, for reference, 'all_max' over all incident edges)
- Node label = 1 iff the agent has at least one malicious outgoing edge (attacker).
- Node-level AUROC / AP vs that label -> directly comparable with G-Safeguard.

Records env/hardware/timestamps like the other runners.
"""
import argparse, json, math, time, os, socket
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from model.tgn import TGN
from utils.utils import get_neighbor_finder, RandEdgeSampler
from utils.data_processing import get_data, compute_time_statistics


def main():
    p = argparse.ArgumentParser("LAMD edge->agent aggregation AUROC")
    p.add_argument('--data', default='mas_mmlu')
    p.add_argument('--prefix', default='abl_full_20261012_072657')
    p.add_argument('--ckpt', default=None, help='default saved_models/{prefix}-{data}.pth')
    p.add_argument('--induct', type=float, default=0.2)
    p.add_argument('--val_time', type=float, default=None)
    p.add_argument('--test_time', type=float, default=None)
    p.add_argument('--n', type=int, default=1000000)
    p.add_argument('--bs', type=int, default=200)
    p.add_argument('--n_degree', type=int, default=10)
    p.add_argument('--n_head', type=int, default=2)
    p.add_argument('--n_layer', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--edge_aware_decoder', action='store_true')
    p.add_argument('--use_memory', action='store_true', help='must match the trained checkpoint')
    p.add_argument('--node_embedding', action='store_true', help='must match the trained checkpoint')
    p.add_argument('--log', default=None)
    args = p.parse_args()

    # ---- logging (env/hardware/timestamp) ----
    logf = args.log or f"exp_logs/lamd_agent_auroc_{time.strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs('exp_logs', exist_ok=True)
    def log(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line)
        with open(logf, 'a') as f:
            f.write(line + "\n")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log(f"host={socket.gethostname()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
        f"torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"data={args.data} prefix={args.prefix} induct={args.induct}")

    # ---- data ----
    class L:
        def info(self, m): log(m)
    node_features, edge_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data = get_data(
            args.data, args.induct, args.n,
            different_new_nodes_between_val_and_test=False, randomize_features=False, logger=L(),
            val_time=args.val_time, test_time=args.test_time)

    full_ngh_finder = get_neighbor_finder(full_data, False, use_direction=True)
    mean_src, std_src, mean_dst, std_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)

    # ---- model (matches the 'full' supervised config) ----
    tgn = TGN(neighbor_finder=full_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=args.n_layer, n_heads=args.n_head, dropout=0.1, use_memory=args.use_memory,
              message_dimension=100, memory_dimension=10, memory_update_at_start=True,
              embedding_module_type="graph_attention", message_function="identity",
              aggregator_type="last", memory_updater_type="gru",
              n_neighbors=args.n_degree,
              mean_time_shift_src=mean_src, std_time_shift_src=std_src,
              mean_time_shift_dst=mean_dst, std_time_shift_dst=std_dst,
              one_hot_dir=True, dir_gnn=True, use_transformer=True,
              edge_aware_decoder=args.edge_aware_decoder,
              node_embedding=args.node_embedding).to(device)

    ckpt = args.ckpt or f"./saved_models/{args.prefix}-{args.data}.pth"
    tgn.load_state_dict(torch.load(ckpt, map_location=device))
    tgn.eval()
    log(f"loaded ckpt {ckpt}")

    # ---- inference over test_data (time-ordered streaming, memory updated) ----
    src_all, score_all, label_all = [], [], []
    n = len(test_data.sources)
    with torch.no_grad():
        for k in range(0, n, args.bs):
            s = k; e = min(n, k + args.bs)
            sb = test_data.sources[s:e]; db = test_data.destinations[s:e]
            tb = test_data.timestamps[s:e]; eb = test_data.edge_idxs[s:e]
            pos_prob, _ = tgn.compute_edge_probabilities(sb, db, db, tb, eb, args.n_degree)
            src_all.append(np.asarray(sb)); score_all.append(pos_prob.cpu().numpy().reshape(-1))
            label_all.append(np.asarray(test_data.labels[s:e]))
    src = np.concatenate(src_all); score = np.concatenate(score_all); elabel = np.concatenate(label_all).astype(int)

    # ---- aggregate per source agent ----
    def agg(mode):
        acc = {}
        for a, sc, lb in zip(src, score, elabel):
            a = int(a)
            if a not in acc:
                acc[a] = [sc, sc, 0]  # max, sum, count
            else:
                acc[a][0] = max(acc[a][0], sc)
                acc[a][1] += sc
                acc[a][2] += 1
        nodes = sorted(acc)
        sc_max = np.array([acc[a][0] for a in nodes])
        sc_mean = np.array([acc[a][1] / acc[a][2] for a in nodes])
        # node label: 1 if any malicious outgoing edge
        nlabel = np.zeros(len(nodes), dtype=int)
        idx = {a: i for i, a in enumerate(nodes)}
        for a, lb in zip(src, elabel):
            if lb == 1:
                nlabel[idx[int(a)]] = 1
        return nodes, sc_max, sc_mean, nlabel

    nodes, sc_max, sc_mean, nlabel = agg('out')
    res = {"n_agents": len(nodes), "n_attacker_agents": int(nlabel.sum())}
    for name, s in [("out_max", sc_max), ("out_mean", sc_mean)]:
        if nlabel.max() == nlabel.min():
            res[name] = {"auc": float('nan'), "ap": float('nan')}
        else:
            res[name] = {"auc": float(roc_auc_score(nlabel, s)),
                         "ap": float(average_precision_score(nlabel, s))}
    log("RESULT " + json.dumps(res))
    log(f"saved log -> {logf}")


if __name__ == "__main__":
    main()
