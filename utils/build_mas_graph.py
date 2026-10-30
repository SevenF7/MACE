"""Build a LAMD-readable temporal directed graph from multi-agent (MAS) dialogues.

Output strictly follows the contract consumed by ``utils/preprocess_data.py``:
  data/{name}.csv  -> header line (skipped by preprocess) + rows ``u,i,ts,label,f0,...``
  data/{name}_conv.npy -> int array, index = REINDEXED node id (orig+1), value = conversation id.
                          Index 0 is unused (-1) to mirror preprocess_data's +1 reindex.

Message = one directed edge sender->receiver. Message text embedding goes into the
EDGE feature columns (node features stay zeros(.,10) as produced by preprocess_data,
because the model hardwires TransformerEncoderLayer(d_model=10) / TimeEncode(n_node_features)).

Two front-ends:
  --source toy          : no-API synthetic generator (for smoke tests)
  --source blindguard    : parse a BlindGuard-style dialogue JSON (finalized on real data)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def embed_texts(texts, dim):
  """Return (len(texts), D) float32 embeddings.

  Prefers sentence-transformers (D=384). Falls back to a deterministic hash-based
  pseudo-embedding of width ``dim`` so smoke tests need no network / API / heavy deps.
  """
  try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(list(texts), show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)
  except Exception:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for k, t in enumerate(texts):
      h = hashlib.sha256(t.encode("utf-8")).digest()
      reps = (dim // len(h)) + 1
      vals = np.frombuffer((h * reps)[:dim], dtype=np.uint8).astype(np.float32)
      out[k] = (vals / 255.0) * 2.0 - 1.0
    return out


def _receivers(topology, a, agents):
  """Directed receivers of agent ``a`` under a given topology."""
  n = len(agents)
  if topology == "chain":
    return [a + 1] if a + 1 < n else []
  if topology == "star":
    return [0] if a != 0 else [j for j in agents if j != 0]
  if topology == "tree":
    kids = [2 * a + 1, 2 * a + 2]
    return [k for k in kids if k < n]
  # random
  others = [j for j in agents if j != a]
  if not others:
    return []
  rng = np.random.RandomState(a + 7)
  k = max(1, rng.randint(1, max(2, len(others))))
  return list(rng.choice(others, size=min(k, len(others)), replace=False))
def gen_toy(n_conv, n_agent, rounds, seed):
  """Synthesize MAS conversations. ~80% of conversations contain one attacker whose
  outgoing messages are labeled malicious (edge-level label=1)."""
  rng = np.random.RandomState(seed)
  topologies = ["random", "chain", "star", "tree"]
  conversations = []
  for c in range(n_conv):
    topo = topologies[c % len(topologies)]
    agents = list(range(n_agent))
    attacker = int(rng.randint(n_agent)) if rng.rand() < 0.8 else -1
    messages = []
    for r in range(rounds):
      for a in agents:
        for b in _receivers(topo, a, agents):
          mal = 1 if a == attacker else 0
          tag = "ATTACK payload inject" if mal else "benign cooperative msg"
          messages.append((a, b, r, f"conv{c} topo={topo} r{r} {a}->{b} {tag}", mal))
    conversations.append({"conv_id": c, "agents": agents, "messages": messages})
  return conversations


def load_blindguard(json_path, agent_pool=0, seed=0):
  """Parse BlindGuard/G-Safeguard dialogue JSON into the intermediate schema.

  json_path may be a single .json file or a directory (all *.json merged).
  Each conversation dict has: adj_matrix (NxN, adj[j][idx]=1 => directed edge j->idx,
  i.e. idx reads j's utterance), attacker_idxes (node-level ground truth),
  communication_data (list of turns; each turn is a list of [agent_idx, text]).

  Edge construction: for every turn t and every adjacency edge j->idx, emit a directed
  edge carrying agent j's utterance of that turn; label=1 iff j is a compromised agent.

  agent_pool>0 remaps each conversation's local agents to a GLOBAL pool of that size, so the
  same agent id recurs across conversations (creating genuine transductive cases) and some
  agents first appear only later (genuine inductive cases). agent_pool=0 keeps per-conversation
  ids (every test-period agent is "new").
  """
  import glob
  if os.path.isdir(json_path):
    files = sorted(f for f in glob.glob(os.path.join(json_path, "*.json"))
                   if os.path.basename(f) != "dataset.json")
  else:
    files = [json_path]
  assert files, f"no json found at {json_path}"

  rng = np.random.RandomState(seed)
  conversations = []
  gcid = 0  # globally-unique conversation id across files
  for fp in files:
    with open(fp) as fh:
      data = json.load(fh)
    for conv in data:
      adj = conv["adj_matrix"]
      n = len(adj)
      attackers = set(int(a) for a in conv.get("attacker_idxes", []))
      if agent_pool > 0:
        pool_ids = rng.choice(agent_pool, size=n, replace=False)
        local2global = {i: int(pool_ids[i]) for i in range(n)}
      elif conv.get("global_agents") is not None:
        gmap = conv["global_agents"]
        local2global = {i: int(gmap[i]) for i in range(n)}
      else:
        local2global = {i: i for i in range(n)}
      messages = []
      for t, turn in enumerate(conv["communication_data"]):
        resp = {}
        for entry in turn:
          resp[int(entry[0])] = entry[1] if isinstance(entry[1], str) else str(entry[1])
        for j in range(n):
          if j not in resp:
            continue
          for idx in range(n):
            if adj[j][idx] == 1:
              messages.append((local2global[j], local2global[idx], t, resp[j],
                               1 if j in attackers else 0))
      if messages:
        agents = sorted(set(local2global.values()))
        conversations.append({"conv_id": gcid, "agents": agents, "messages": messages})
        gcid += 1
  n_mal = sum(1 for c in conversations for m in c["messages"] if m[4] == 1)
  n_edge = sum(len(c["messages"]) for c in conversations)
  n_uniq = len(set(a for c in conversations for a in c["agents"]))
  # Multi-session data: order by session then chain so all benign sessions precede attack
  # sessions on the global time axis (lets the default time split separate them).
  if conversations and "session_id" in conversations[0]:
    conversations.sort(key=lambda c: (c.get("session_id", 0), c.get("chain_id", 0)))
  print(f"[load_blindguard] {len(files)} file(s) -> {len(conversations)} conversations, "
        f"{n_edge} edges, {n_mal} malicious edges, {n_uniq} unique global agents "
        f"(agent_pool={agent_pool})")
  return conversations


def build(conversations, name, feat_dim, share_agents=False):
  node_map, node_conv = {}, {}
  next_id = 0
  rows, texts = [], []
  BIG = 100000  # per-conversation time offset to keep a global ascending ts
  for ci, conv in enumerate(conversations):
    cid = conv["conv_id"]
    for a in conv["agents"]:
      key = a if share_agents else (cid, a)
      if key not in node_map:
        node_map[key] = next_id
        node_conv[next_id] = cid
        next_id += 1
    for (s, d, r, text, mal) in conv["messages"]:
      u = node_map[s if share_agents else (cid, s)]
      i = node_map[d if share_agents else (cid, d)]
      rows.append((u, i, ci * BIG + r, int(mal), len(texts)))
      texts.append(text)
  rows.sort(key=lambda x: x[2])  # guarantee ts ascending

  feats = embed_texts(texts, feat_dim)
  d = feats.shape[1]
  data_dir = Path("data")
  data_dir.mkdir(parents=True, exist_ok=True)
  out_csv = data_dir / f"{name}.csv"
  with open(out_csv, "w") as f:
    f.write("u,i,ts,label," + ",".join(f"f{k}" for k in range(d)) + "\n")
    for (u, i, ts, mal, ti) in rows:
      fe = ",".join(f"{x:.6f}" for x in feats[ti])
      f.write(f"{u},{i},{ts},{float(mal)},{fe}\n")

  # Sidecar aligned to preprocess_data's non-bipartite reindex (orig id + 1); index 0 unused.
  n_nodes = next_id
  conv_arr = np.full((n_nodes + 1,), -1, dtype=np.int64)
  for nid, cid in node_conv.items():
    conv_arr[nid + 1] = cid
  np.save(data_dir / f"{name}_conv.npy", conv_arr)

  n_pos = sum(1 for r in rows if r[3] == 1)
  print(f"[build_mas_graph] {out_csv}: {len(rows)} edges, {n_nodes} nodes, "
        f"{len(conversations)} conversations, edge_feat_dim={d}, malicious_edges={n_pos}")
  print(f"[build_mas_graph] sidecar: {data_dir / (name + '_conv.npy')} (len={n_nodes + 1})")
  print(f"[build_mas_graph] next: python utils/preprocess_data.py --data {name}  (do NOT pass --bipartite)")


def main():
  p = argparse.ArgumentParser("Build LAMD temporal graph from MAS dialogues")
  p.add_argument("--source", choices=["toy", "blindguard"], default="toy")
  p.add_argument("--json", type=str, default=None, help="BlindGuard dialogue JSON dir (train)")
  p.add_argument("--json_test", type=str, default=None,
                 help="optional second dir (test), appended after --json so its conversations get later timestamps")
  p.add_argument("--name", type=str, required=True, help="dataset name -> data/{name}.csv")
  p.add_argument("--n_conv", type=int, default=40)
  p.add_argument("--n_agent", type=int, default=8)
  p.add_argument("--rounds", type=int, default=4)
  p.add_argument("--feat_dim", type=int, default=64, help="edge feature width for the hash fallback")
  p.add_argument("--agent_pool", type=int, default=0,
                 help=">0: remap agents to a global pool of this size (enables true inductive split)")
  p.add_argument("--share_agents", action="store_true",
                 help="treat agent ids as already-global (persistent across conversations), e.g. multi-session data")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  if args.source == "toy":
    conversations = gen_toy(args.n_conv, args.n_agent, args.rounds, args.seed)
  else:
    assert args.json, "--json is required for --source blindguard"
    conversations = load_blindguard(args.json, agent_pool=args.agent_pool, seed=args.seed)
    if args.json_test:
      test_convs = load_blindguard(args.json_test, agent_pool=args.agent_pool, seed=args.seed)
      off = (max(c["conv_id"] for c in conversations) + 1) if conversations else 0
      for c in test_convs:
        c["conv_id"] += off
      conversations += test_convs
      print(f"[build_mas_graph] appended test dir: total {len(conversations)} conversations")
  build(conversations, args.name, args.feat_dim,
        share_agents=(args.agent_pool > 0 or args.share_agents))


if __name__ == "__main__":
  main()
