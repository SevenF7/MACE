import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


class ConvNegEdgeSampler(object):
  """Within-conversation negative sampler for MAS graphs.

  Each MAS conversation is a disjoint subgraph; sampling a global random destination would
  almost always yield a trivial cross-conversation non-edge, weakening the contrastive signal.
  This sampler draws the negative destination from the SAME conversation as the positive source.

  node_conv: int array indexed by (reindexed) node id -> conversation id (data/{name}_conv.npy).
  sample(size, src_batch) requires the positive sources to know each edge's conversation.
  """
  def __init__(self, src_list, dst_list, node_conv, seed=None):
    self.seed = seed
    self.node_conv = np.asarray(node_conv)
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    dst_convs = self.node_conv[self.dst_list]
    self.conv_to_dsts = {}
    for conv in np.unique(dst_convs):
      self.conv_to_dsts[int(conv)] = self.dst_list[dst_convs == conv]
    self.random_state = np.random.RandomState(seed) if seed is not None else None

  def _randint(self, high):
    if self.random_state is not None:
      return self.random_state.randint(0, high)
    return np.random.randint(0, high)

  def sample(self, size, src_batch):
    assert len(src_batch) == size
    src_index = (self.random_state.randint(0, len(self.src_list), size)
                 if self.random_state is not None
                 else np.random.randint(0, len(self.src_list), size))
    neg = np.empty(size, dtype=self.dst_list.dtype)
    for k in range(size):
      conv = int(self.node_conv[src_batch[k]])
      cand = self.conv_to_dsts.get(conv, self.dst_list)
      if len(cand) == 0:
        cand = self.dst_list
      neg[k] = cand[self._randint(len(cand))]
    return self.src_list[src_index], neg

  def reset_random_state(self):
    if self.seed is not None:
      self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None, use_direction=False):
  """
  Build temporal neighbor finder from data.
  use_direction: if True, store and return edge direction (src_indicator) per neighbor:
    True = current node is source (outgoing edge), False = current node is destination (incoming).
    Required by the orientation code and the dual-branch attention. Default False keeps undirected neighbours.
  """
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    if use_direction:
      adj_list[source].append((destination, edge_idx, timestamp, True))   # from source: outgoing
      adj_list[destination].append((source, edge_idx, timestamp, False))  # from dest: incoming
    else:
      adj_list[source].append((destination, edge_idx, timestamp))
      adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform, use_direction=use_direction)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None, use_direction=False):
    self.use_direction = use_direction
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []
    if use_direction:
      self.node_to_src_indicator = []  # True = outgoing from current node, False = incoming

    for neighbors in adj_list:
      # Neighbors: list of (neighbor, edge_idx, timestamp) or (neighbor, edge_idx, timestamp, is_outgoing)
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
      if use_direction:
        self.node_to_src_indicator.append(np.array([x[3] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    When use_direction is True, returns 4 arrays: neighbors, edge_idxs, timestamps, src_indicator.
    Otherwise returns 3 arrays: neighbors, edge_idxs, timestamps.
    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    if self.use_direction:
      return (self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i],
              self.node_to_edge_timestamps[src_idx][:i], self.node_to_src_indicator[src_idx][:i])
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    When use_direction is True, returns (neighbors, edge_idxs, edge_times, src_indicator) with
    src_indicator[i,j] True if the edge from source_nodes[i] to neighbors[i,j] is outgoing (current node is source).
    Otherwise returns (neighbors, edge_idxs, edge_times) for backward compatibility.
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.float32)
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
    if self.use_direction:
      src_indicator = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=bool)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      result = self.find_before(source_node, timestamp)
      if self.use_direction:
        source_neighbors, source_edge_idxs, source_edge_times, source_src_indicator = result
      else:
        source_neighbors, source_edge_idxs, source_edge_times = result

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]
          if self.use_direction:
            src_indicator[i, :] = source_src_indicator[sampled_idx]
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
          if self.use_direction:
            src_indicator[i, :] = src_indicator[i, :][pos]
        else:
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]
          if self.use_direction:
            source_src_indicator = source_src_indicator[-n_neighbors:]
          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)
          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
          if self.use_direction:
            src_indicator[i, n_neighbors - len(source_src_indicator):] = source_src_indicator

    if self.use_direction:
      return neighbors, edge_idxs, edge_times, src_indicator
    return neighbors, edge_idxs, edge_times