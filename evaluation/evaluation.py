import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, recall_score, precision_score, roc_curve


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_recall, val_precision, val_fp, val_fn, val_tp, val_tn = [], [], [], [], [], [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      new_pred_score = []
      #print("########################")
      #print("Pred_score")
      #print(pred_score.shape)
      #print(np.squeeze(pred_score).shape)
      #print(pred_score[:30])
      
      #print("########################")
      #print(pred_score[:30])

      #print("########################")
      #print("true_label")
      #print(true_label.shape)
      #print(true_label[:30])

      # val_ap.append(average_precision_score(true_label, pred_score))
      # val_auc.append(roc_auc_score(true_label, pred_score))
      
      ### Joseph: Added roc_cuve to get the optial threshold ####
      fpr, tpr, thresholds = roc_curve(true_label, pred_score)
      
      gmean = np.sqrt(tpr * (1 - fpr))

      # Find the optimal threshold
      index = np.argmax(gmean)
      #### Use the  "thresholdOpt" for the best threshold
      thresholdOpt = round(thresholds[index], ndigits = 4)
      gmeanOpt = round(gmean[index], ndigits = 4)
      
      # print("########################")
      # print("thresholdOpt ", thresholdOpt)
      # print("gmeanOpt ", gmeanOpt)
      # print("our threshold 0.50")
      # print("########################")

      for i in range(len(pred_score)):
        ## Modified byb Joseph
        # if pred_score[i] <= 0.50:
        if pred_score[i] < thresholdOpt:
          pred_score[i] = 0
        else:
          pred_score[i] = 1
      
      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))  
      
      #print(recall_score(true_label, pred_score))
      val_recall.append(recall_score(true_label, pred_score))
      val_precision.append(precision_score(true_label, pred_score))

      

      # conf_mtrx = confusion_matrix(true_label, pred_score)
      # val_fp.append(conf_mtrx.sum(axis=0) - np.diag(conf_mtrx))
      # val_fn.append(conf_mtrx.sum(axis=1) - np.diag(conf_mtrx))
      # val_tp.append(np.diag(conf_mtrx))
      # val_tn.append(conf_mtrx.values.sum() - (FP + FN + TP))
      
      tn, fp, fn, tp = confusion_matrix(true_label, pred_score).ravel()
      val_fp.append(fp)
      val_fn.append(fn)
      val_tp.append(tp)
      val_tn.append(tn)

  #return np.mean(val_ap), np.mean(val_auc), np.mean(val_fp), np.mean(val_fn), np.mean(val_tp), np.mean(val_tn)
  # return np.mean(val_ap), np.mean(val_auc), np.mean(val_recall), np.mean(val_precision)
  return np.mean(val_ap), np.mean(val_auc), np.mean(val_recall), np.mean(val_precision), np.mean(val_fp), np.mean(val_fn), np.mean(val_tp), np.mean(val_tn)


def eval_anomaly_detection(model, data, n_neighbors, batch_size=200, score_mode='low_prob'):
  """Edge/message-level anomaly detection using the ground-truth malicious labels.

  score_mode='low_prob' : anomaly = 1 - p(edge)  (self-supervised link-prediction readout)
  score_mode='direct'   : anomaly = p(edge)       (supervised edge-classifier readout)
  Scores ONLY the real edges and compares against data.labels (1 = malicious edge).
  Batches are processed in time order so streaming memory stays consistent.
  """
  with torch.no_grad():
    model = model.eval()
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    all_scores, all_labels = [], []
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)
      if e_idx <= s_idx:
        continue
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

      # negatives are unused by the positive branch; pass destinations as a dummy
      pos_prob, _ = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                     destinations_batch, timestamps_batch,
                                                     edge_idxs_batch, n_neighbors)
      p = pos_prob.cpu().numpy().reshape(-1)
      all_scores.append((1.0 - p) if score_mode == 'low_prob' else p)
      all_labels.append(np.asarray(data.labels[s_idx:e_idx]).reshape(-1))

  scores = np.concatenate(all_scores) if all_scores else np.array([])
  labels = np.concatenate(all_labels).astype(int) if all_labels else np.array([], dtype=int)
  result = {"n": int(len(labels)), "n_pos": int(labels.sum()) if len(labels) else 0}
  if len(labels) == 0 or labels.max() == labels.min():
    result.update({"ap": float("nan"), "auc": float("nan"),
                   "precision": float("nan"), "recall": float("nan")})
    return result
  result["ap"] = float(average_precision_score(labels, scores))
  result["auc"] = float(roc_auc_score(labels, scores))
  fpr, tpr, thresholds = roc_curve(labels, scores)
  gmean = np.sqrt(tpr * (1 - fpr))
  opt = thresholds[int(np.argmax(gmean))]
  pred = (scores >= opt).astype(int)
  result["threshold"] = float(opt)
  result["precision"] = float(precision_score(labels, pred, zero_division=0))
  result["recall"] = float(recall_score(labels, pred, zero_division=0))
  return result


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
