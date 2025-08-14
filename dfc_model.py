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
    def __init__(self,
                 input_dim,            # raw node feature size (e.g., number of regions)
                 hidden_dim=128,
                 output_dim=256,
                 num_classes=3,
                 dropout=0.5,
                 use_topk_pooling=True,
                 topk_ratio=0.5,
                 layer_type="GCN",     # "GCN", "GAT", or "GraphSAGE"
                 temporal_aggregation="mean",  # "mean", "max", or "gru"
                 num_layers=3,
                 activation='relu'):
        super(DynamicGraphNeuralNetwork, self).__init__()

        self.use_topk_pooling = use_topk_pooling
        self.temporal_aggregation = temporal_aggregation
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Activation function
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu
        }
        self.activation_fn = activation_map.get(activation, F.relu)

        # Project raw input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers + GraphNorm
        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim

            if layer_type == "GCN":
                conv = GCNConv(in_dim, out_dim)
            elif layer_type == "GAT":
                conv = GATConv(in_dim, out_dim)
            elif layer_type == "GraphSAGE":
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown GNN layer type: {layer_type}")

            self.convs.append(conv)
            self.gns.append(GraphNorm(out_dim))

        self.dropout = nn.Dropout(p=dropout)

        # Optional TopKPooling layers
        if use_topk_pooling:
            safe_ratio = max(0.3, min(1.0, topk_ratio))
            self.topk_pools = nn.ModuleList([
                TopKPooling(hidden_dim if i < num_layers - 1 else output_dim, ratio=safe_ratio)
                for i in range(num_layers)
            ])
        else:
            self.topk_pools = None

        # Temporal aggregation
        self.temporal_dim = 2 * output_dim  # mean + max pooling
        if temporal_aggregation == "gru":
            self.gru = nn.GRU(input_size=self.temporal_dim,
                              hidden_size=self.temporal_dim,
                              batch_first=True)
        else:
            self.gru = None

        # Final classifier
        self.classifier = nn.Linear(self.temporal_dim, num_classes)

    def forward(self, x, edge_index, batch, time_features=None):
        """
        Process a single graph and return embeddings.
        This method is compatible with TemporalDataLoader.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
            batch: batch assignment for nodes
            time_features: optional time features (not used in this encoder)

        Returns:
            embeddings: tensor of shape (batch_size, 2 * output_dim)
        """
        # Project input features
        x = self.input_proj(x)

        # Apply GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.gns[i](x, batch)
            x = self.activation_fn(x)
            x = self.dropout(x)

            if self.use_topk_pooling and i < len(self.topk_pools):
                x, edge_index, _, batch, _, _ = self.topk_pools[i](x, edge_index, batch=batch)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_repr = torch.cat([x_mean, x_max], dim=1)  # (batch_size, 2 * output_dim)

        return graph_repr
    
    def forward_sequence(self, x_seq, edge_index_seq, batch_seq):
        """
        Process a sequence of graphs for classification.
        Original method for processing temporal sequences.
        
        Args:
            x_seq: list of tensors, each (num_nodes_t, input_dim)
            edge_index_seq: list of edge_index tensors per time
            batch_seq: list of batch tensors per time

        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        time_outputs = []

        for t in range(len(x_seq)):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            batch = batch_seq[t]

            # Use the single graph forward method
            graph_repr = self.forward(x, edge_index, batch)
            time_outputs.append(graph_repr)

        # Shape: (batch_size, time_steps, 2 * output_dim)
        time_outputs = torch.stack(time_outputs, dim=1)

        # Temporal aggregation
        if self.temporal_aggregation == "mean":
            out = time_outputs.mean(dim=1)
        elif self.temporal_aggregation == "max":
            out, _ = time_outputs.max(dim=1)
        elif self.temporal_aggregation == "gru":
            _, h = self.gru(time_outputs)
            out = h.squeeze(0)
        else:
            raise ValueError(f"Unsupported temporal aggregation: {self.temporal_aggregation}")

        # Final classification
        logits = self.classifier(out)
        return logits
    
    def load_state_dict_flexible(self, state_dict):
        """Load state dict with flexibility for missing/extra keys"""
        model_dict = self.state_dict()
        # Filter out keys that don't match
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict, strict=False)
