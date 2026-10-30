import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction, eval_anomaly_detection
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, ConvNegEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

### Argument and global variables
parser = argparse.ArgumentParser('LAMD self-supervised training')
parser.add_argument('--seed', type=int, default=0, help='Model-init / data-order seed')
parser.add_argument('--induct', type=float, default=0.9)
parser.add_argument('--n', type=int, default=1000000)
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. auth or pivoting)',
                    default='auth')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
# parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
#                                                                 'each user')
parser.add_argument('--memory_dim', type=int, default=10, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--use_direction', action='store_true',
                    help='[Phase 0] Store and use edge direction (src_indicator) in neighbor finder for OHD-TGN/DIR-TGN. Default: off, original undirected neighbor behavior.')
parser.add_argument('--one_hot_dir', action='store_true',
                    help='[Phase 1 OHD-TGN] Add learned direction encoding to edge features (sum to ê_ij). Implies --use_direction. Default: off.')
parser.add_argument('--dir_gnn', action='store_true',
                    help='[Phase 2 DIR-TGN] Two GNNs for out-/in-edges and fuse (z = alpha*z_out + (1-alpha)*z_in). Implies --use_direction. Default: off.')
parser.add_argument('--no_transformer', action='store_true',
                    help='Ablation: skip the TransformerEncoderLayer on node embeddings (identity path).')
parser.add_argument('--conv_neg', action='store_true',
                    help='[MAS] Use within-conversation negative sampling (reads data/{data}_conv.npy). Default: off.')
parser.add_argument('--mas_eval', action='store_true',
                    help='[MAS] Additionally run edge-level anomaly detection vs ground-truth labels. Default: off.')
parser.add_argument('--supervised_edge', action='store_true',
                    help='[MAS] Supervise the decoder with edge malicious labels (p(edge)=P(malicious)) '
                         'instead of link-prediction negatives. Encoder unchanged. Implies --mas_eval.')
parser.add_argument('--val_time', type=float, default=None,
                    help='[MAS] Explicit validation split time (overrides the 0.70 time quantile).')
parser.add_argument('--test_time', type=float, default=None,
                    help='[MAS] Explicit test split time (overrides the 0.85 time quantile). Pin to the '
                         'train/test conversation boundary to align with the node-level baselines.')
parser.add_argument('--edge_aware_decoder', action='store_true',
                    help='[MAS] Content-aware decoder: score edges from (src_emb, dst_emb, edge_features) '
                         'so the current message content directly drives P(malicious).')
parser.add_argument('--node_embedding', action='store_true',
                    help='[MAS] Add a learnable per-agent embedding to the (all-zero) node features.')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
induct = args.induct
n = args.n
if args.supervised_edge:
  args.mas_eval = True  # supervised edge classification is evaluated via the MAS anomaly metric

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
Path("log/").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='log/{}.log'.format(str(time.time())),
                    filemode='a',                    
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI')
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, induct, n,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, logger = logger,
                              val_time=args.val_time, test_time=args.test_time)

# Direction: OHD-TGN and DIR-TGN require direction in neighbor finder
use_direction_flag = args.use_direction or args.one_hot_dir or args.dir_gnn
# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform, use_direction=use_direction_flag)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform, use_direction=use_direction_flag)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
if args.conv_neg:
  node_conv = np.load('./data/{}_conv.npy'.format(DATA))
  train_rand_sampler = ConvNegEdgeSampler(train_data.sources, train_data.destinations, node_conv)
else:
  node_conv = None
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
logger.info("$$$ ==> {}".format(device_string))
device = torch.device(device_string)
logger.info("### ==> {}".format(device))

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            one_hot_dir=args.one_hot_dir,
            dir_gnn=args.dir_gnn,
            use_transformer=not args.no_transformer,
            edge_aware_decoder=args.edge_aware_decoder,
            node_embedding=args.node_embedding)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    # prog = tqdm(desc='Batches parsed', total=num_batch)
    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      # prog.update(num_batch-k)
      if (k%10000 == 0):
          logger.info('Batch number: {}/{}'.format(k, num_batch))
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        if args.conv_neg:
          _, negatives_batch = train_rand_sampler.sample(size, sources_batch)
        else:
          _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        if args.supervised_edge:
          # Supervise the real-edge probability to equal P(malicious) using edge labels.
          # p(edge) becomes the malicious score; negatives are unused.
          with torch.no_grad():
            edge_label = torch.tensor(train_data.labels[start_idx:end_idx], dtype=torch.float, device=device)
          loss += criterion(pos_prob.squeeze(), edge_label)
        else:
          loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()


    val_ap, val_auc, val_recall, val_precision, val_fp, val_fn, val_tp, val_tn = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)


    #val_ap, val_auc, FP, FN, TP, TN = eval_edge_prediction(model=tgn,
    #                                                        negative_edge_sampler=val_rand_sampler,
    #                                                        data=val_data,
    #                                                        n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc, nn_val_recall, nn_val_precision, nn_val_fp, nn_val_fn, nn_val_tp, nn_val_tn = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    if args.mas_eval:
      # Edge-level anomaly detection vs ground-truth labels. Restore the end-of-training
      # memory first (its last_update precedes val timestamps) so re-streaming val edges
      # does not move memory "into the past"; then restore the post-val state.
      if USE_MEMORY:
        _mas_bk = tgn.memory.backup_memory()
        tgn.memory.restore_memory(train_memory_backup)
      mas_val = eval_anomaly_detection(tgn, val_data, NUM_NEIGHBORS,
                                       score_mode=('direct' if args.supervised_edge else 'low_prob'))
      if USE_MEMORY:
        tgn.memory.restore_memory(_mas_bk)
      logger.info('MAS val anomaly (edge-level): {}'.format(mas_val))

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info(
      'val recall: {}, new node val recall: {}'.format(val_recall, nn_val_recall))
    logger.info(
      'val precision: {}, new node val precision: {}'.format(val_precision, nn_val_precision))
    
    logger.info(
      'val fp: {}, new node val fp: {}'.format(val_fp, nn_val_fp))
    logger.info(
      'val fn: {}, new node val fn: {}'.format(val_fn, nn_val_fn))
    logger.info(
      'val tp: {}, new node val tp: {}'.format(val_tp, nn_val_tp))
    logger.info(
      'val tn: {}, new node val tn: {}'.format(val_tn, nn_val_tn))

    # Early stopping. For supervised edge classification, select on MAS val AUC (detection
    # quality) rather than link-prediction AP.
    stop_metric = val_ap
    if args.supervised_edge and args.mas_eval and not math.isnan(mas_val.get('auc', float('nan'))):
      stop_metric = mas_val['auc']
    if early_stopper.early_stop_check(stop_metric):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()


  ### Test
  logger.info('Started Testing')
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(test_auc, test_ap, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(nn_test_auc, nn_test_ap, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))

  if args.mas_eval:
    # Restore end-of-validation memory so re-streaming test edges does not move memory "into the past".
    if USE_MEMORY:
      _mas_bk = tgn.memory.backup_memory()
      tgn.memory.restore_memory(val_memory_backup)
    mas_test = eval_anomaly_detection(tgn, test_data, NUM_NEIGHBORS,
                                      score_mode=('direct' if args.supervised_edge else 'low_prob'))
    if USE_MEMORY:
      tgn.memory.restore_memory(val_memory_backup)
    mas_nn_test = eval_anomaly_detection(tgn, new_node_test_data, NUM_NEIGHBORS,
                                         score_mode=('direct' if args.supervised_edge else 'low_prob'))
    if USE_MEMORY:
      tgn.memory.restore_memory(_mas_bk)
    logger.info('MAS test anomaly (edge-level, old nodes): {}'.format(mas_test))
    logger.info('MAS test anomaly (edge-level, new nodes): {}'.format(mas_nn_test))
    print('MAS test anomaly (edge-level, old nodes):', mas_test)
    print('MAS test anomaly (edge-level, new nodes):', mas_nn_test)





  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
