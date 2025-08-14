import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import os
import copy
import numpy as np
import pandas as pd
from DFC_ADNIDataset import DFC_ADNIDataset
from model import GraphNeuralNetwork
from dfc_model import DynamicGraphNeuralNetwork
from TemporalPredictor import TemporalTabGNNClassifier
from GRUPredictor import GRUPredictor
from RNNPredictor import RNNPredictor
from TemporalDataLoader import TemporalDataLoader
from FocalLoss import FocalLoss
from conversion_analyzer import analyze_conversion_predictions, print_conversion_accuracy_report, aggregate_conversion_results
import random

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for temporal sequences (subjects per batch)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--focal_alpha', type=float, default=0.90, help='focal loss alpha (weight for minority class)')
parser.add_argument('--focal_gamma', type=float, default=3.0, help='focal loss gamma (focusing parameter)')
parser.add_argument('--label_smoothing', type=float, default=0.05, help='label smoothing factor')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--minority_focus_epochs', type=int, default=20, help='epochs to focus on minority class')
parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='LSTM hidden dimension')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of LSTM layers')
parser.add_argument('--lstm_bidirectional', type=bool, default=True, help='use bidirectional LSTM')
parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds (set to 1 for single fold)')
parser.add_argument('--model_type', type=str, default='LSTM', help='Which RNN Based Model is being utilized RNN, GRU, LSTM')
parser.add_argument('--pretrain_encoder', action='store_true', help='whether to pretrain GNN encoder using GraphCL')
parser.add_argument('--pretrain_epochs', type=int, default=50, help='number of epochs for GNN self-supervised pretraining')
parser.add_argument('--freeze_encoder', action='store_true', help='whether to freeze GNN encoder during temporal training')
parser.add_argument('--use_pretrained', action='store_true', help='use an existing pretrained encoder if it is on disk')
parser.add_argument('--use_topk_pooling', action='store_true', default=True, help='use TopK pooling instead of global pooling')
parser.add_argument('--topk_ratio', type=float, default=0.3, help='TopK pooling ratio (fraction of nodes to keep)')
parser.add_argument('--layer_type', type=str, default="GraphSAGE", help='GNN layer type: GCN, GAT, or GraphSAGE')
parser.add_argument('--gnn_hidden_dim', type=int, default=256, help='GNN hidden dimension size')
parser.add_argument('--gnn_num_layers', type=int, default=2, help='number of GNN layers (2-5)')
parser.add_argument('--gnn_activation', type=str, default='elu', help='GNN activation function: relu, leaky_relu, elu, gelu')
parser.add_argument('--use_time_features', action='store_true', help='use temporal gap features for time-aware prediction')
parser.add_argument('--exclude_target_visit', action='store_true', help='exclude target visit from input sequences (prevent data leakage)')
parser.add_argument('--time_normalization', type=str, default='log', help='time normalization method: log, minmax, buckets, raw')
parser.add_argument('--single_visit_horizon', type=int, default=6, help='default prediction horizon (months) for single-visit subjects')
parser.add_argument('--temporal_aggregation', type=str, default='mean', choices=['mean', 'max', 'last'],
                    help='Temporal aggregation strategy for sequence embeddings')
opt = parser.parse_args()

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)

if opt.use_topk_pooling and (opt.topk_ratio <= 0.0 or opt.topk_ratio > 1.0):
    print(f"Warning: Invalid topk_ratio {opt.topk_ratio}. Setting to 0.5 (keep 50% of nodes)")
    opt.topk_ratio = 0.5

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

fold_results = {
    'test_acc': [],
    'balanced_acc': [],
    'minority_f1': [],
    'test_auc': [],
    'train_acc': [],
    'balanced_train_acc': []
}

fold_conversion_results = []

###################
# GNN PRETRAINING UTILITIES
###################

def graph_augmentation(data, aug_type="drop_node", aug_ratio=0.2, seed=None):
    """Simple graph augmentation for GraphCL-style pretraining."""
    data = data.clone()
    device = data.x.device
    
    # Set seed for reproducible augmentation
    if seed is not None:
        torch.manual_seed(seed)

    if aug_type == "drop_node":
        num_nodes = data.x.size(0)  # Use actual number of nodes from feature tensor
        node_mask = torch.rand(num_nodes, device=device) > aug_ratio
        data.x = data.x[node_mask]

        # Remap node indices
        new_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        new_idx[node_mask] = torch.arange(node_mask.sum(), device=device)

        # Filter and remap edges
        edge_index = data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_index = new_idx[edge_index]
        data.edge_index = edge_index

    elif aug_type == "drop_edge":
        edge_mask = torch.rand(data.edge_index.size(1), device=device) > aug_ratio
        data.edge_index = data.edge_index[:, edge_mask]
    else:
        raise ValueError("Unsupported augmentation type")

    return data

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.exp(torch.matmul(reps, reps.T) / temperature)

    batch_size = z1.size(0)
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)

    mask = (~torch.eye(2 * batch_size, dtype=torch.bool)).to(z1.device)
    sim_sum = sim_matrix.masked_select(mask).view(2 * batch_size, -1).sum(dim=1)

    loss = -torch.log(pos_sim / (sim_sum[:batch_size] + sim_sum[batch_size:]))
    return loss.mean()


def pretrain_graph_encoder(encoder, dataset, device, epochs=50, batch_size=32, lr=1e-3):
    """GraphCL-style self-supervised pretraining for the GNN encoder."""
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    # Create seeded generator for reproducible data loading
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=0, generator=g)

    print(f"Pretraining GNN Encoder for {epochs} epochs (GraphCL)")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            data_list = batch.to_data_list()

            # Use different seeds for different augmentations to ensure variety
            batch1 = [graph_augmentation(d, "drop_node", 0.2, seed=42+i) for i, d in enumerate(data_list)]
            batch2 = [graph_augmentation(d, "drop_edge", 0.2, seed=100+i) for i, d in enumerate(data_list)]

            batch1 = Batch.from_data_list(batch1)
            batch2 = Batch.from_data_list(batch2)

            z1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
            z2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Pretrain Epoch {epoch+1:03d}] Loss: {avg_loss:.4f}")

    print("GNN Encoder Pretraining Complete.")

################## 
# LOAD DATASET
##################

dataset = DFC_ADNIDataset(
    root="/media/volume/ADNI-Data/git/TabGNN/FinalDeliverables/data",
    var_name="dynamic_fc"  # updated variable within your .npz structures
)
print(f"Loaded {len(dataset)} DFC graph samples")

# Clean up any infinite values in node features (just in case)
dataset.data.x[torch.isinf(dataset.data.x)] = 0

# Build subject-level mapping for temporal sequences
if not hasattr(dataset, 'subject_graph_dict') or dataset.subject_graph_dict is None:
    mapping = {}
    for data in dataset:
        sid = getattr(data, 'subj_id', None)
        if sid is None:
            continue
        # Extract base subject ID (e.g., from 'sub-002S0413_run-01' to '002S0413')
        if '_run' in sid:
            base_sid = sid.split('_run')[0]
            if 'sub-' in base_sid:
                base_sid = base_sid.replace('sub-', '')
        else:
            base_sid = sid.replace('sub-', '') if 'sub-' in sid else sid
        mapping.setdefault(base_sid, []).append(data)
    dataset.subject_graph_dict = mapping

print(f"Built subject graph mapping: {len(dataset.subject_graph_dict)} subjects")

# Trim per-subject visits if exceeding limit

# K-fold cross-validation splits at the subject level
def get_kfold_splits(ds, num_folds=5, seed=42):
    subj_labels = {
        sid: visits[0].y.item()
        for sid, visits in ds.subject_graph_dict.items()
        if visits and hasattr(visits[0], 'y')
    }

    subjects = list(subj_labels.keys())
    labels = [subj_labels[s] for s in subjects]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, labels)):
        train_subj = [subjects[i] for i in train_idx]
        test_subj = [subjects[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]

        tr_idx, val_idx = train_test_split(
            np.arange(len(train_subj)),
            test_size=0.2,
            random_state=seed + fold,
            stratify=train_labels
        )

        splits.append({
            'train_subj': [train_subj[i] for i in tr_idx],
            'val_subj':   [train_subj[i] for i in val_idx],
            'test_subj':  test_subj
        })
    return splits

fold_splits = get_kfold_splits(dataset, num_folds=opt.num_folds)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
set_random_seeds(42)

encoder = DynamicGraphNeuralNetwork(
    input_dim= dataset.data.x.size(-1),  # dynamic input dim from dataset
    hidden_dim=opt.gnn_hidden_dim,
    output_dim=256,
    num_classes=2,  # Adjust as appropriate
    use_topk_pooling=opt.use_topk_pooling,
    topk_ratio=opt.topk_ratio,
    layer_type=opt.layer_type,
    temporal_aggregation=opt.temporal_aggregation,
    num_layers=opt.gnn_num_layers,
    activation=opt.gnn_activation
).to(device)

pretrained_path = os.path.join(opt.save_path, 'pretrained_gnn_encoder.pth')
set_random_seeds(42)

if getattr(opt, 'pretrain_encoder', False):
    if getattr(opt, 'use_pretrained', False) and os.path.exists(pretrained_path):
        #encoder.load_state_dict(torch.load(pretrained_path))
        print("Loaded pretrained encoder.")
    else:
        print("Pretraining GNN encoder...")
        pretrain_graph_encoder(encoder, dataset, device, epochs=getattr(opt, 'pretrain_epochs', 50))
        torch.save(encoder.state_dict(), pretrained_path)
        print("Pretrained encoder saved.")
elif os.path.exists(pretrained_path):
    #encoder.load_state_dict(torch.load(pretrained_path))
    print("Loaded frozen pretrained encoder.")
else:
    print("No pretrained weights found; training from scratch.")

subject_id_to_indices = {}
for idx, data in enumerate(dataset):
    subj_id = getattr(data, 'subj_id', None)
    if subj_id is None:
        print(f"Warning: Data at index {idx} has no subj_id")
        continue
    # Map both full ID and base ID to indices
    if subj_id not in subject_id_to_indices:
        subject_id_to_indices[subj_id] = []
    subject_id_to_indices[subj_id].append(idx)
    
    # Extract base subject ID properly (e.g., from 'sub-002S0413_run-01' to '002S0413')
    if '_run' in subj_id:
        base_sid = subj_id.split('_run')[0]
        if 'sub-' in base_sid:
            base_sid = base_sid.replace('sub-', '')
    else:
        base_sid = subj_id.replace('sub-', '') if 'sub-' in subj_id else subj_id
    
    if base_sid != subj_id:
        if base_sid not in subject_id_to_indices:
            subject_id_to_indices[base_sid] = []
        subject_id_to_indices[base_sid].append(idx)

print(f"Built subject ID to indices mapping for {len(subject_id_to_indices)} unique IDs")

# === Define this helper function BEFORE the fold loop ===
def idx_for_subj_list(subj_list):
    indices = []
    missing_subjects = []
    for subj in subj_list:
        subj_indices = subject_id_to_indices.get(subj, [])
        if not subj_indices:
            missing_subjects.append(subj)
        indices.extend(subj_indices)
    if missing_subjects:
        print(f"Warning: {len(missing_subjects)} subjects not found in indices mapping")
    return indices

# === Fold loop ===
for fold, split in enumerate(fold_splits, start=1):
    print(f"\n=== Fold {fold}/{opt.num_folds} ===")
    tr_subj, val_subj, te_subj = split.values()

    train_idx = idx_for_subj_list(tr_subj)
    val_idx = idx_for_subj_list(val_subj)
    test_idx = idx_for_subj_list(te_subj)
    print(f"[DEBUG] Number of training indices: {len(train_idx)}")
    print(f"[DEBUG] Training indices: {train_idx}")
    fold_encoder = copy.deepcopy(encoder).to(device)
    if opt.freeze_encoder:
        for p in fold_encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen.")

    # Temporal loaders
    train_loader = TemporalDataLoader(dataset, train_idx, fold_encoder, device,
                                      batch_size=opt.batch_size, shuffle=True, seed=42,
                                      exclude_target_visit=opt.exclude_target_visit,
                                      time_normalization=opt.time_normalization,
                                      single_visit_horizon=opt.single_visit_horizon)
    val_loader = TemporalDataLoader(dataset, val_idx, fold_encoder, device,
                                    batch_size=opt.batch_size, shuffle=False, seed=42,
                                    exclude_target_visit=opt.exclude_target_visit,
                                    time_normalization=opt.time_normalization,
                                    single_visit_horizon=opt.single_visit_horizon)
    test_loader = TemporalDataLoader(dataset, test_idx, fold_encoder, device,
                                     batch_size=opt.batch_size, shuffle=False, seed=42,
                                     exclude_target_visit=opt.exclude_target_visit,
                                     time_normalization=opt.time_normalization,
                                     single_visit_horizon=opt.single_visit_horizon)

    print(f"Batches per epoch: {len(train_loader)}")
    
    if len(train_loader) == 0:
        print(f"ERROR: Training loader is empty for fold {fold}!")
        print(f"Training subjects: {len(tr_subj)}, indices: {len(train_idx)}")
        print(f"Validation subjects: {len(val_subj)}, indices: {len(val_idx)}")
        print(f"Test subjects: {len(te_subj)}, indices: {len(test_idx)}")
        continue

    # Setup classifier based on temporal model type
    set_random_seeds(42)
    graph_emb_dim = fold_encoder.output_dim * 2  # mean+max pooling
    if opt.model_type == "LSTM":
        classifier = TemporalTabGNNClassifier(graph_emb_dim, 0, opt.lstm_hidden_dim,
                                              opt.lstm_num_layers, dropout=0.45,
                                              bidirectional=opt.lstm_bidirectional,
                                              num_classes=2).to(device)
    elif opt.model_type == "GRU":
        classifier = GRUPredictor(graph_emb_dim, 0, opt.lstm_hidden_dim,
                                  opt.lstm_num_layers, dropout=0.45,
                                  bidirectional=False, num_classes=2).to(device)
    else:
        classifier = RNNPredictor(graph_emb_dim, 0, opt.lstm_hidden_dim,
                                  opt.lstm_num_layers, dropout=0.45,
                                  bidirectional=False, num_classes=2).to(device)

    set_random_seeds(42)
    optimizer = torch.optim.Adam([
        {'params': fold_encoder.parameters(), 'lr': opt.lr * 0.5},
        {'params': classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(alpha=opt.focal_alpha, gamma=opt.focal_gamma, label_smoothing=opt.label_smoothing)

    class_labels = np.array([data.y.item() for data in dataset])
    print(f"Using Focal Loss with alpha={opt.focal_alpha}, gamma={opt.focal_gamma}")
    print(f"Class distribution: {np.bincount(class_labels)}")

    ###################
    # TRAINING UTILITIES
    ###################

    def evaluate_detailed(loader, return_probs=False):
        """Comprehensive evaluation with detailed metrics for temporal sequences."""
        fold_encoder.eval()
        classifier.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                graph_seq = batch['graph_seq']
                lengths = batch['lengths'] 
                labels = batch['labels']
                time_gaps = batch.get('time_gaps', None)
                
                logits = classifier(graph_seq, None, lengths)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        minority_precision = precision[1] if len(precision) > 1 else 0
        minority_recall = recall[1] if len(recall) > 1 else 0
        minority_f1 = f1[1] if len(f1) > 1 else 0
        
        balanced_acc = np.mean([recall[0] if len(recall) > 0 else 0, 
                                recall[1] if len(recall) > 1 else 0])
        
        probs_positive = np.array(all_probs)[:, 1]
        auc_score = roc_auc_score(all_targets, probs_positive)
        
        result = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'minority_precision': minority_precision,
            'minority_recall': minority_recall,
            'minority_f1': minority_f1,
            'auc': auc_score,
            'predictions': all_preds,
            'targets': all_targets,
            'unique_preds': len(set(all_preds))
        }

        if return_probs:
            result['probabilities'] = all_probs
        
        return result

    def evaluate_by_horizon(loader):
        """Evaluate model performance grouped by prediction time horizon."""
        if not opt.use_time_features:
            return None
            
        fold_encoder.eval()
        classifier.eval()
        
        # Define time horizon buckets (in normalized log scale)
        # log(1 + months/12): 0-6m→0.35, 6-12m→0.69, 12-24m→1.1, 24m+→>1.1
        horizons = {
            '0-6m': {'range': (0, 0.5), 'preds': [], 'targets': [], 'probs': []},
            '6-12m': {'range': (0.5, 0.8), 'preds': [], 'targets': [], 'probs': []},
            '12-24m': {'range': (0.8, 1.2), 'preds': [], 'targets': [], 'probs': []},
            '24m+': {'range': (1.2, float('inf')), 'preds': [], 'targets': [], 'probs': []}
        }
        
        with torch.no_grad():
            for batch in loader:
                graph_seq = batch['graph_seq']
                lengths = batch['lengths']
                labels = batch['labels']
                time_gaps = batch.get('time_gaps', None)
                
                if time_gaps is None:
                    continue
                    
                logits = classifier(graph_seq, None, lengths)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Group predictions by time horizon
                for i in range(len(time_gaps)):
                    gap = time_gaps[i].item()
                    pred = preds[i].item()
                    target = labels[i].item()
                    prob = probs[i].cpu().numpy()
                    
                    # Find appropriate bucket
                    for horizon_name, horizon_data in horizons.items():
                        if horizon_data['range'][0] <= gap < horizon_data['range'][1]:
                            horizon_data['preds'].append(pred)
                            horizon_data['targets'].append(target)
                            horizon_data['probs'].append(prob)
                            break
        
        # Calculate metrics for each horizon
        results = {}
        for horizon_name, horizon_data in horizons.items():
            if len(horizon_data['preds']) > 0:
                preds = np.array(horizon_data['preds'])
                targets = np.array(horizon_data['targets'])
                probs = np.array(horizon_data['probs'])
                
                accuracy = np.mean(preds == targets)
                
                # Calculate per-class metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, preds, average=None, zero_division=0
                )
                
                # Calculate AUC if both classes present
                try:
                    auc = roc_auc_score(targets, probs[:, 1])
                except:
                    auc = 0.0
                
                results[horizon_name] = {
                    'count': len(preds),
                    'accuracy': accuracy,
                    'minority_f1': f1[1] if len(f1) > 1 else 0,
                    'auc': auc
                }
            else:
                results[horizon_name] = {
                    'count': 0,
                    'accuracy': 0,
                    'minority_f1': 0,
                    'auc': 0
                }
        
        return results

    def minority_class_forcing_loss(logits, targets, epoch):
        """
        Additional loss term to force minority class prediction in early epochs.
        """
        if epoch > opt.minority_focus_epochs:
            return torch.tensor(0.0, device=logits.device)
        
        # Much gentler approach - only apply to minority class samples and reduce weight
        minority_mask = (targets == 1)
        if minority_mask.sum() > 0:
            minority_logits = logits[minority_mask]
            # Encourage class 1 prediction for minority samples
            minority_loss = F.cross_entropy(minority_logits, torch.ones(minority_mask.sum(), dtype=torch.long, device=logits.device))
            # Greatly reduce the forcing weight - make it much smaller
            forcing_weight = 0.1 * (opt.minority_focus_epochs - epoch) / opt.minority_focus_epochs
            return forcing_weight * minority_loss
        
        return torch.tensor(0.0, device=logits.device)


    ###################
    # TRAINING LOOP
    ###################

    print(f"\nStarting {opt.model_type} Training for Fold {fold + 1}")
    
    best_auc = 0.0
    best_minority_f1 = 0.0
    best_balanced_acc = 0.0
    patience = 20
    patience_counter = 0
    best_model_state = None
    train_results = {'accuracy': 0.0, 'balanced_accuracy': 0.0}

    for epoch in range(1, opt.n_epochs + 1):
        
        fold_encoder.train()
        classifier.train()
        
        total_loss = 0
        total_focal_loss = 0
        total_forcing_loss = 0
        correct = 0
        total = 0
        class_0_preds = 0
        class_1_preds = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            graph_seq = batch['graph_seq']
            lengths = batch['lengths'] 
            labels = batch['labels']
            time_gaps = batch.get('time_gaps', None)
            
            logits = classifier(graph_seq, None, lengths)
            
            # Main loss
            loss = criterion(logits, labels)
            
            forcing_loss = minority_class_forcing_loss(logits, labels, epoch)

            total_batch_loss = loss + forcing_loss
            
            total_batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(fold_encoder.parameters()) + list(classifier.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_focal_loss += loss.item()
            total_forcing_loss += forcing_loss.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            class_0_preds += (preds == 0).sum().item()
            class_1_preds += (preds == 1).sum().item()
            
        
        scheduler.step()
        
        if len(train_loader) > 0:
            avg_loss = total_loss / len(train_loader)
        else:
            avg_loss = 0.0
            print(f"Warning: No batches in train_loader for epoch {epoch}")
        
        train_acc = correct / total if total > 0 else 0.0
        
        val_results = evaluate_detailed(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Model saving based on AUC
        current_auc = val_results['auc']
        
        if (val_results['unique_preds'] > 1 and current_auc > best_auc) or \
        (val_results['unique_preds'] > 1 and best_auc == 0):
            best_auc = val_results['auc']
            best_minority_f1 = val_results['minority_f1']
            best_balanced_acc = val_results['balanced_accuracy']
            patience_counter = 0
            
            best_model_state = {
                'encoder': fold_encoder.state_dict().copy(),
                'classifier': classifier.state_dict().copy(),
                'epoch': epoch,
                'val_results': val_results
            }
            
            # Evaluate current training performance for the best model
            train_results = evaluate_detailed(train_loader)
            
            if opt.save_model:
                torch.save(best_model_state, os.path.join(opt.save_path, f'best_model_fold{fold}.pth'))
            
            print(f"New best model saved (AUC: {best_auc:.3f}, Balanced Acc: {best_balanced_acc:.3f})")
        else:
            patience_counter += 1

    # Load best model for fold evaluation
    if best_model_state is not None:
        fold_encoder.load_state_dict_flexible(best_model_state['encoder'])
        classifier.load_state_dict(best_model_state['classifier'])
        print(f"\nLoaded best model from epoch {best_model_state['epoch']}")
    else:
        print("\nWarning: No valid model found, using current state")

    ###################
    # Test Set Evaluation
    ###################

    print("\n" + "="*50)
    print(f"Test Set Evaluation - Fold {fold + 1}")

    test_results = evaluate_detailed(test_loader, return_probs=True)
    
    print(f"Test Loss: {test_results['loss']:.4f} | Balanced Accuracy: {test_results['balanced_accuracy']:.3f}")
    
    # Prediction distribution
    test_pred_counts = {i: test_results['predictions'].count(i) for i in set(test_results['predictions'])}
    print(f"Prediction Distribution: {test_pred_counts}")

    print("\nClassification Report:")
    print(classification_report(test_results['targets'], test_results['predictions'],
                                target_names=['Stable', 'Converter'],
                                zero_division=0))
    
    # Horizon-based evaluation only IF using time features
    if opt.use_time_features:
        print("\n" + "="*50)
        print("By horizon:")
        horizon_results = evaluate_by_horizon(test_loader)
        if horizon_results:
            for horizon, metrics in horizon_results.items():
                if metrics['count'] > 0:
                    print(f"{horizon}: n={metrics['count']}, Acc={metrics['accuracy']:.3f}, F1={metrics['minority_f1']:.3f}")
                else:
                    print(f"{horizon}: none")

    # Conversion-specific accuracy analysis
    label_csv_path = os.path.join("/media/volume/ADNI-Data/git/TabGNN/FinalDeliverables/data", "TADPOLE_Simplified.csv")
    conversion_results = analyze_conversion_predictions(
        test_subjects, 
        test_results['predictions'], 
        test_results['targets'], 
        label_csv_path
    )
    print_conversion_accuracy_report(conversion_results)
    fold_conversion_results.append(conversion_results)

    fold_results['test_acc'].append(test_results['accuracy'])
    fold_results['balanced_acc'].append(test_results['balanced_accuracy'])
    fold_results['minority_f1'].append(test_results['minority_f1'])
    fold_results['test_auc'].append(test_results['auc'])
    fold_results['train_acc'].append(train_results['accuracy'])
    fold_results['balanced_train_acc'].append(train_results['balanced_accuracy'])

print("\nCross-Validation Summary:")
for metric, values in fold_results.items():
    if values:  # Avoid empty lists
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean:.3f} ± {std:.3f}")
    else:
        print(f"{metric}: No data collected")

# Aggregated conversion-specific accuracy analysis
if fold_conversion_results:
    print("\n" + "="*60)
    print("AGGREGATED CONVERSION ACCURACY ACROSS ALL FOLDS")
    print("="*60)
    aggregated_conversion_results = aggregate_conversion_results(fold_conversion_results)
    print_conversion_accuracy_report(aggregated_conversion_results)