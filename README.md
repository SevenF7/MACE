# MACE

Code for **MACE: Securing LLM-Based Agent Service Compositions through Content-Aware Temporal Graph Anomaly Detection**.

MACE scores each message between LLM agent services as an edge of a directed temporal graph, using a temporal graph attention backbone with direction-aware dual-branch aggregation, an intra-batch relational adapter (IBRA) and a content-aware edge decoder.

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data format

The dataset itself is not distributed in this repository. Place each dataset `<name>` in `./data/` as three files:

| File | Content |
|---|---|
| `ml_<name>.csv` | one row per edge with columns `u` (sender id), `i` (receiver id), `ts` (timestamp), `label` (1 = malicious, 0 = benign), `idx` (edge index, starting at 1) |
| `ml_<name>.npy` | edge features, shape `(num_edges + 1, d_e)`; row `idx` holds the frozen sentence embedding of message `idx` (`d_e = 384` for all-MiniLM-L6-v2); row 0 is padding |
| `ml_<name>_node.npy` | node features, shape `(num_nodes + 1, 10)` |

`utils/build_mas_graph.py` and `utils/preprocess_data.py` show how a message log is converted into this format. Edges must be sorted by `ts`. The chain-level split of the paper is selected with `--val_time` and `--test_time` (the timestamps at which the validation and test chains start).

## Training and evaluation

Full model with the reference configuration of the paper (K = 10, L = 1, learning rate 1e-4, dropout 0.1, batch size 200, 80 epochs):

```bash
python train_self_supervised.py --data <name> --supervised_edge --mas_eval \
    --induct 0 --val_time <VAL_TS> --test_time <TEST_TS> \
    --n_degree 10 --n_layer 1 --lr 1e-4 --drop_out 0.1 --bs 200 \
    --n_epoch 80 --patience 80 --seed 1 --gpu 0 --prefix mace \
    --use_memory --one_hot_dir --dir_gnn --edge_aware_decoder
```

Checkpoints are written to `saved_models/` and `saved_checkpoints/`, logs to `log/`.

Ablation variants change only the model flags:

| Variant | Flags |
|---|---|
| Full model | `--use_memory --one_hot_dir --dir_gnn --edge_aware_decoder` |
| w/o IBRA | full model + `--no_transformer` |
| w/o direction components | `--use_memory --edge_aware_decoder` |
| orientation code only | `--use_memory --one_hot_dir --edge_aware_decoder` |
| dual branches only | `--use_memory --dir_gnn --edge_aware_decoder` |
| stock decoder | `--use_memory --one_hot_dir --dir_gnn` |
| stock decoder + node-identity embedding | `--use_memory --one_hot_dir --dir_gnn --node_embedding` |

Per-edge test scores (AUC / AP) of a trained model:

```bash
python eval_lamd_edge_scores.py --data <name> --prefix mace --induct 0 \
    --val_time <VAL_TS> --test_time <TEST_TS> --use_memory --edge_aware_decoder --tag MACE__<name>
python summarize_scores.py
```

Agent-level read-out (AUROC / AUPRC of max / mean aggregated edge scores):

```bash
python eval_lamd_agent_auroc.py --data <name> --prefix mace --induct 0.2 \
    --use_memory --edge_aware_decoder
```

`--use_memory`, `--edge_aware_decoder` and `--node_embedding` must match the trained checkpoint. The agent-level results use an inductive node split (`--induct 0.2`).

## License

All rights reserved.
