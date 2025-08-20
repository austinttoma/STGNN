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

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pretrain_encoder_supervised(encoder, dataset, device, epochs=50, batch_size=32, lr=1e-3):
    """Pretrain the GNN encoder using supervised learning on graph classification."""
    
    # Split dataset
    labels = dataset.data.y.cpu().numpy()
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
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
    
    # Pretrain with supervision
    encoder = pretrain_encoder_supervised(encoder, dataset, device, epochs=100, batch_size=32, lr=1e-3)
    
    # Save the pretrained encoder
    save_path = "./model/pretrained_gnn_encoder.pth"
    torch.save(encoder.state_dict(), save_path)
    print(f"\nPretrained encoder saved to {save_path}")
    
    # Check file size
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"File size: {file_size:.1f} KB")

if __name__ == "__main__":
    main()