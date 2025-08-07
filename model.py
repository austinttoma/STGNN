import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    TopKPooling,
)
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=256, dropout=0.5, 
                 use_topk_pooling=True, topk_ratio=0.5, layer_type="GCN", 
                 num_layers=3, activation='relu', use_time_features=False):
        super(GraphNeuralNetwork, self).__init__()

        self.use_topk_pooling = use_topk_pooling
        self.topk_ratio = topk_ratio
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_time_features = use_time_features
        
        # Set up activation function
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu
        }
        self.activation_fn = activation_map.get(activation, F.relu)
        
        # Build dynamic layer architecture
        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()
        
        # Create layers dynamically
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> hidden_dim
                in_dim = input_dim
                out_dim = hidden_dim
            elif i == num_layers - 1:
                # Last layer: hidden_dim -> output_dim
                in_dim = hidden_dim
                out_dim = output_dim
            else:
                # Middle layers: hidden_dim -> hidden_dim
                in_dim = hidden_dim
                out_dim = hidden_dim
            
            # Create layer based on type
            if layer_type == "GCN":
                conv = GCNConv(in_dim, out_dim)
            elif layer_type == "GAT":
                conv = GATConv(in_dim, out_dim)
            elif layer_type == "GraphSAGE":
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.convs.append(conv)
            self.gns.append(GraphNorm(out_dim))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # TopK pooling layers for hierarchical pooling with safeguards
        if use_topk_pooling:
            # Ensure minimum 30% of nodes are retained at each layer to prevent empty graphs
            safe_ratio = max(0.3, min(1.0, topk_ratio))
            self.topk_pools = nn.ModuleList()
            
            for i in range(num_layers):
                if i == num_layers - 1:
                    # Last layer uses output_dim
                    pool_dim = output_dim
                else:
                    # Other layers use hidden_dim
                    pool_dim = hidden_dim
                
                self.topk_pools.append(TopKPooling(pool_dim, ratio=safe_ratio, min_score=None))
            
            print(f"TopK pooling initialized with {num_layers} layers, safe ratio: {safe_ratio} (minimum 30%)")
        
        # Traditional pooling ops as fallback
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        
        # Time feature projection layer if using time features
        if self.use_time_features:
            # Project time feature to match output dimension
            # Note: final GNN output is actually output_dim*2 due to mean+max pooling
            final_output_dim = output_dim * 2
            # Much smaller time projection to prevent dominance over graph features
            time_dim = 32  # Small dimension for time features
            self.time_projection = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, time_dim)
            )
            # Weighted fusion layer - graph features get more weight than time features
            self.fusion_layer = nn.Sequential(
                nn.Linear(final_output_dim + time_dim, final_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, x, edge_index, batch, time_to_predict=None):
        if self.use_topk_pooling:
            graph_embedding = self._forward_with_topk(x, edge_index, batch)
        else:
            graph_embedding = self._forward_traditional(x, edge_index, batch)
        
        # If using time features, combine them with graph embedding
        if self.use_time_features and time_to_predict is not None:
            # time_to_predict is [B] or [B, 1], ensure it's [B, 1]
            if len(time_to_predict.shape) == 1:
                time_to_predict = time_to_predict.unsqueeze(1)
            
            # Project time feature to smaller dimension
            time_embedding = self.time_projection(time_to_predict)  # [B, time_dim=32]
            
            # Concatenate graph (512D) + time (32D) = 544D total
            combined = torch.cat([graph_embedding, time_embedding], dim=1)  # [B, final_output_dim + time_dim]
            final_embedding = self.fusion_layer(combined)  # [B, final_output_dim]
            
            return final_embedding
        else:
            return graph_embedding
    
    def _forward_with_topk(self, x, edge_index, batch):
        # Dynamic forward pass through all layers with TopK pooling
        for i in range(self.num_layers):
            # Apply convolution
            x = self.convs[i](x, edge_index)
            x = self.gns[i](x, batch)
            x = self.activation_fn(x)
            x = self.dropout(x)
            
            # Apply TopK pooling after each layer
            x, edge_index, _, batch, perm, score = self.topk_pools[i](x, edge_index, batch=batch)
        
        # Global pooling on the remaining top nodes
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        
        # Combine pooled representations
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x
    
    def _forward_traditional(self, x, edge_index, batch):
        # Dynamic forward pass through all layers without TopK pooling
        for i in range(self.num_layers):
            # Apply convolution
            x = self.convs[i](x, edge_index)
            x = self.gns[i](x, batch)
            x = self.activation_fn(x)
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