import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    TopKPooling
)

class DynamicGraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=256, dropout=0.5,
                 use_topk_pooling=True, topk_ratio=0.5, layer_type="GCN",
                 temporal_aggregation="mean", num_layers=3, activation='relu'):
        super(DynamicGraphNeuralNetwork, self).__init__()
        
        self.use_topk_pooling = use_topk_pooling
        self.temporal_aggregation = temporal_aggregation
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
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
        
        self.dropout = nn.Dropout(p=dropout)
        
        # TopK pooling layers for hierarchical pooling
        if use_topk_pooling:
            safe_ratio = max(0.3, min(1.0, topk_ratio))
            self.topk_pools = nn.ModuleList()
            
            for i in range(num_layers):
                if i == num_layers - 1:
                    # Last layer uses output_dim
                    pool_dim = output_dim
                else:
                    # Other layers use hidden_dim
                    pool_dim = hidden_dim
                
                self.topk_pools.append(TopKPooling(pool_dim, ratio=safe_ratio))
        
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool

    def forward(self, x_seq, edge_index_seq, batch_seq):
        """
        x_seq: list of length T with shape (N_i, F)
        edge_index_seq: list of length T with edge_index tensors
        batch_seq: list of length T with batch tensors
        """
        time_outputs = []

        for t in range(len(x_seq)):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            batch = batch_seq[t]

            # Dynamic forward pass through all layers
            for i in range(self.num_layers):
                # Apply convolution
                x = self.convs[i](x, edge_index)
                x = self.gns[i](x, batch)
                x = self.activation_fn(x)
                x = self.dropout(x)
                
                # Apply TopK pooling if enabled
                if self.use_topk_pooling:
                    x, edge_index, _, batch, _, _ = self.topk_pools[i](x, edge_index, batch=batch)

            x_mean = self.pool_mean(x, batch)
            x_max = self.pool_max(x, batch)
            graph_repr = torch.cat([x_mean, x_max], dim=1)  # (batch_size, 2 * output_dim)

            time_outputs.append(graph_repr)

        time_outputs = torch.stack(time_outputs, dim=1)  # (batch_size, T, 2*output_dim)

        # Temporal aggregation
        if self.temporal_aggregation == "mean":
            out = time_outputs.mean(dim=1)
        elif self.temporal_aggregation == "max":
            out, _ = time_outputs.max(dim=1)
        elif self.temporal_aggregation == "gru":
            if not hasattr(self, 'gru'):
                self.gru = nn.GRU(input_size=2 * output_dim, hidden_size=2 * output_dim,
                                  batch_first=True)
            _, h = self.gru(time_outputs)
            out = h.squeeze(0)
        else:
            raise ValueError("Unsupported temporal aggregation method")

        return out