#!/usr/bin/env python3
"""
Supervised pretraining for the GNN encoder.
This is more reliable than contrastive learning and avoids representation collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GraphNeuralNetwork
from FC_ADNIDataset import FC_ADNIDataset
from sklearn.model_selection import StratifiedKFold

def get_main_test_subjects(dataset, num_folds=5, seed=42, fold_to_exclude=0):
    """Get test subjects from a specific fold to avoid data leakage."""
    # Replicate main.py's subject splitting logic
    subject_graph_dict = {}
    for data in dataset:
        sid = getattr(data, 'subj_id', None)
        if sid is None:
            continue
        base_subject_id = sid.split('_run')[0] if '_run' in sid else sid
        subject_graph_dict.setdefault(base_subject_id, []).append(data)

    subject_labels = {}
    for subj_id, graphs in subject_graph_dict.items():
        if graphs and hasattr(graphs[0], 'y'):
            subject_labels[subj_id] = graphs[0].y.item()

    subjects = list(subject_labels.keys())
    labels = [subject_labels[s] for s in subjects]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Only exclude test subjects from fold 0 to preserve enough data for pretraining
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(subjects, labels)):
        if fold_idx == fold_to_exclude:
            test_subjects = [subjects[i] for i in test_idx]
            return test_subjects
    
    return []

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pretrain_encoder_supervised(encoder, dataset, device, epochs=50, batch_size=32, lr=1e-3, exclude_subjects=None):
    """Pretrain the GNN encoder using supervised learning on graph classification."""
    
    # Create subject-level split to avoid data leakage with main.py
    subject_graph_dict = {}
    for i, data in enumerate(dataset):
        sid = getattr(data, 'subj_id', None)
        if sid is None:
            continue
        base_subject_id = sid.split('_run')[0] if '_run' in sid else sid
        if base_subject_id not in subject_graph_dict:
            subject_graph_dict[base_subject_id] = []
        subject_graph_dict[base_subject_id].append(i)
    
    # Get subject labels
    subject_labels = {}
    for subj_id, visit_indices in subject_graph_dict.items():
        # Use label from first visit (all visits for same subject should have same label)
        subject_labels[subj_id] = dataset[visit_indices[0]].y.item()
    
    # Split by subjects, not visits
    subjects = list(subject_labels.keys())
    
    # Exclude subjects that will be used in main.py evaluation to prevent data leakage
    if exclude_subjects is not None:
        excluded_count = 0
        for excluded_subj in exclude_subjects:
            if excluded_subj in subjects:
                subjects.remove(excluded_subj)
                excluded_count += 1
        print(f"Excluded {excluded_count} subjects from pretraining to prevent data leakage")
    
    labels = [subject_labels[s] for s in subjects]
    
    # Use different random state to avoid correlation with main.py splits
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=0.2, stratify=labels, random_state=123  # Different from main.py's 42
    )
    
    # Get visit indices for each subject split
    train_idx = []
    val_idx = []
    for subj in train_subjects:
        train_idx.extend(subject_graph_dict[subj])
    for subj in val_subjects:
        val_idx.extend(subject_graph_dict[subj])
    
    print(f"Subject-level split: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects")
    print(f"Visit-level split: {len(train_idx)} train visits, {len(val_idx)} val visits")
    
    # Create data loaders
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
    
    # Add a simple classifier head for pretraining
    classifier = nn.Sequential(
        nn.Linear(512 if encoder.use_topk_pooling else 256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    ).to(device)
    
    # Optimizer for both encoder and classifier
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), 
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    
    print(f"Supervised Pretraining for {epochs} epochs")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"Train subjects: {train_subjects[:5]}...")  # Show first 5 for verification
    print(f"Val subjects: {val_subjects[:5]}...")      # Show first 5 for verification
    
    best_val_acc = 0
    best_encoder_state = None
    
    for epoch in range(epochs):
        # Training
        encoder.train()
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            embeddings = encoder(batch.x, batch.edge_index, batch.batch)
            logits = classifier(embeddings)
            loss = F.cross_entropy(logits, batch.y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        # Validation
        encoder.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = encoder(batch.x, batch.edge_index, batch.batch)
                logits = classifier(embeddings)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder_state = encoder.state_dict().copy()
            print(f"[Epoch {epoch+1:03d}/{epochs}] Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f} (best)")
        else:
            print(f"[Epoch {epoch+1:03d}/{epochs}] Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    # Load best model
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        print(f"\nPretraining complete. Best validation accuracy: {best_val_acc:.3f}")
    
    return encoder

def main():
    # Set seeds
    set_random_seeds(42)
    
    # Load dataset
    dataset = FC_ADNIDataset(
        root="/media/volume/ADNI-Data/git/TabGNN/FinalDeliverables/data",
        var_name="fc_matrix"
    )
    print(f"Loaded {len(dataset)} fMRI FC graph samples")
    dataset.data.y = dataset.data.y.squeeze()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create encoder with correct architecture
    encoder = GraphNeuralNetwork(
        input_dim=100,
        hidden_dim=256,
        output_dim=256,
        dropout=0.2,
        use_topk_pooling=True,
        topk_ratio=0.3,
        layer_type="GraphSAGE",
        num_layers=2,
        activation="elu",
        use_time_features=False
    ).to(device)
    
    print("\nEncoder architecture:")
    print(f"  Input dim: 100")
    print(f"  Hidden dim: 256")
    print(f"  Output dim: 256 (512 with TopK pooling)")
    print(f"  Dropout: 0.2")
    print(f"  Layer type: GraphSAGE")
    print(f"  Num layers: 2")
    print(f"  TopK ratio: 0.3\n")
    
    # Get test subjects from fold 0 of main.py to exclude from pretraining (prevent data leakage)
    main_test_subjects = get_main_test_subjects(dataset, num_folds=5, seed=42, fold_to_exclude=0)
    print(f"Excluding {len(main_test_subjects)} test subjects from main.py fold 0")
    
    # Pretrain with supervision, excluding main.py test subjects
    encoder = pretrain_encoder_supervised(encoder, dataset, device, epochs=50, batch_size=32, lr=1e-3, 
                                        exclude_subjects=main_test_subjects)
    
    # Save the pretrained encoder
    save_path = "./model/pretrained_gnn_encoder.pth"
    torch.save(encoder.state_dict(), save_path)
    print(f"\nPretrained encoder saved to {save_path}")
    
    # Check file size
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"File size: {file_size:.1f} KB")

if __name__ == "__main__":
    main()