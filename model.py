import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    TopKPooling,
)

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=256, dropout=0.5, 
                 use_topk_pooling=True, topk_ratio=0.5):
        super(GraphNeuralNetwork, self).__init__()

        self.use_topk_pooling = use_topk_pooling
        self.topk_ratio = topk_ratio

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Graph normalization layers (per-graph stats)
        self.gn1 = GraphNorm(hidden_dim)
        self.gn2 = GraphNorm(hidden_dim)
        self.gn3 = GraphNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # TopK pooling layers for hierarchical pooling with safeguards
        if use_topk_pooling:
            # Ensure minimum 30% of nodes are retained at each layer to prevent empty graphs
            safe_ratio = max(0.3, min(1.0, topk_ratio))
            self.topk_pool1 = TopKPooling(hidden_dim, ratio=safe_ratio, min_score=None)
            self.topk_pool2 = TopKPooling(hidden_dim, ratio=safe_ratio, min_score=None) 
            self.topk_pool3 = TopKPooling(output_dim, ratio=safe_ratio, min_score=None)
            print(f"TopK pooling initialized with safe ratio: {safe_ratio} (minimum 30%)")
        
        # Traditional pooling ops as fallback
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool

    def forward(self, x, edge_index, batch):
        if self.use_topk_pooling:
            return self._forward_with_topk(x, edge_index, batch)
        else:
            return self._forward_traditional(x, edge_index, batch)
    
    def _forward_with_topk(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.gn1(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # First TopK pooling - keep top nodes
        x, edge_index, _, batch, perm, score = self.topk_pool1(x, edge_index, batch=batch)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.gn2(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second TopK pooling
        x, edge_index, _, batch, perm, score = self.topk_pool2(x, edge_index, batch=batch)
        
        # Third GCN layer  
        x = self.conv3(x, edge_index)
        x = self.gn3(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final TopK pooling
        x, edge_index, _, batch, perm, score = self.topk_pool3(x, edge_index, batch=batch)
        
        # Global pooling on the remaining top nodes
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        
        # Combine pooled representations
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x
    
    def _forward_traditional(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.gn1(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.gn2(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.gn3(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Combine mean and max pooling for richer representation
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x
    
    def load_state_dict_flexible(self, state_dict, strict=False):
        # Get current model's state dict
        model_state_dict = self.state_dict()
        
        # Check for missing TopK pooling keys
        missing_topk_keys = []
        if self.use_topk_pooling:
            topk_keys = ['topk_pool1.select.weight', 'topk_pool2.select.weight', 'topk_pool3.select.weight']
            for key in topk_keys:
                if key not in state_dict and key in model_state_dict:
                    missing_topk_keys.append(key)
        
        if missing_topk_keys:
            print(f"Warning: Loading pretrained model without TopK pooling layers.")
            print(f"Missing keys: {missing_topk_keys}")
            print("TopK pooling layers will be initialized with random weights.")
            
            # Initialize missing keys with current model's initialized weights
            for key in missing_topk_keys:
                state_dict[key] = model_state_dict[key].clone()
        
        # Load the state dict
        return super().load_state_dict(state_dict, strict=strict)