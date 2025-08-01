import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd

def fc_to_graph(fc_matrix, node_features=None, threshold=0.2, subj_id=None):
    """
    Convert a single [N, N] FC matrix into a PyTorch Geometric graph.
    """
    A = fc_matrix.copy()
    N = A.shape[0]

    # Threshold weak connections
    A[np.abs(A) < threshold] = 0

    # Get edges
    edge_index = np.array(np.nonzero(A))
    edge_attr = A[edge_index[0], edge_index[1]]

    # Convert to torch
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)

    # Default node features: identity matrix
    if node_features is None:
        x = torch.eye(N, dtype=torch.float)
    else:
        x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if subj_id is not None:
        data.subj_id = subj_id  # Attach subject ID

    return data

def load_fc_graph_sequences_walk(base_path, threshold=0.2, var_name='fc_matrix'):
    """
    Recursively loads all .npz FC matrix files and converts them into sequences of graph Data objects.
    """
    fc_graphs = {}
    npz_files = []

    print(f"Searching in: {base_path}")

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith("_fc_matrix.npz"):
                npz_files.append(os.path.join(root, file))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No FC .npz files found under {base_path}")

    print(f"Found {len(npz_files)} .npz FC files.")

    for path in tqdm(sorted(npz_files), desc="Converting FCs"):
        try:
            fname = os.path.basename(path)
            subj_id = fname.split('_')[0].replace('sub-', '')

            data = np.load(path)

            if var_name not in data:
                print(f"Warning: '{var_name}' not in {path}. Skipping.")
                continue

            fc_tensor = data[var_name]  # Shape: [T, N, N]

            if subj_id not in fc_graphs:
                fc_graphs[subj_id] = []

            for t in range(fc_tensor.shape[0]):
                mat = fc_tensor[t]
                mat = (mat + mat.T) / 2  # Enforce symmetry
                graph = fc_to_graph(mat, threshold=threshold, subj_id=subj_id)
                fc_graphs[subj_id].append(graph)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return fc_graphs

def load_subject_labels(label_csv_path, label_col='Label_CS_Num'):
    df = pd.read_csv(label_csv_path)
    df['Subject'] = df['Subject'].str.replace('_', '', regex=False)
    return dict(zip(df['Subject'], df[label_col]))

def create_padding_graph(num_nodes, label, subj_id=None):
    x = torch.zeros((num_nodes, num_nodes))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
    if subj_id is not None:
        data.subj_id = subj_id
    return data

def createPadded(fc_graphs):
    padded_fc_graphs = {}
    max_seq_len = max(len(graphs) for graphs in fc_graphs.values())
    print(f"Max sequence length: {max_seq_len}")

    for subj_id, graphs in fc_graphs.items():
        seq_len = len(graphs)
        padded_graphs = graphs.copy()

        if seq_len < max_seq_len:
            num_nodes = graphs[0].x.size(0)
            label = graphs[0].y if hasattr(graphs[0], 'y') else torch.tensor([-1])

            pad_graph = create_padding_graph(num_nodes, label, subj_id=subj_id)
            padded_graphs += [pad_graph] * (max_seq_len - seq_len)

        padded_fc_graphs[subj_id] = padded_graphs

    return padded_fc_graphs

def summarize_patient_graph_dims(padded_graphs):
    summaries = []

    for subj_id, graphs in padded_graphs.items():
        nodes = [g.x.size(0) for g in graphs]
        feats = [g.x.size(1) for g in graphs]
        edges = [g.edge_index.size(1) for g in graphs]
        label = int(graphs[0].y.item()) if hasattr(graphs[0], 'y') else -1

        summaries.append({
            'subject_id': subj_id,
            'num_graphs': len(graphs),
            'avg_nodes': int(np.mean(nodes)),
            'avg_features': int(np.mean(feats)),
            'avg_edges': int(np.mean(edges)),
            'label': label
        })

    return pd.DataFrame(summaries)

def main():
    base_path = "/media/volume/ADNI-Data/git/TabGNN/data/New_FC_Matrices/Updated FC Matrices"
    label_csv_path = "/media/volume/ADNI-Data/git/TabGNN/data/TADPOLE_Simplified.csv"

    label_dict = load_subject_labels(label_csv_path)
    fc_graphs = load_fc_graph_sequences_walk(base_path, threshold=0.2, var_name="arr_0")

    # Add labels
    for subj_id in fc_graphs:
        clean_id = subj_id.replace('_', '')
        label = label_dict.get(clean_id, 0)
        for graph in fc_graphs[subj_id]:
            graph.y = torch.tensor([label], dtype=torch.long)

    print(f"Loaded {len(fc_graphs)} subjects.")
    padded_graphs = createPadded(fc_graphs)

    print("\nSubject Summary:")
    for subj_id, graphs in padded_graphs.items():
        label = int(graphs[0].y.item()) if hasattr(graphs[0], 'y') else -1
        print(f"Subject: {subj_id} | Length: {len(graphs)} | Label: {label}")

    # ➕ Generate and export summary
    df_summary = summarize_patient_graph_dims(padded_graphs)
    print(df_summary.head())
    df_summary.to_csv("data/graph_summary.csv", index=False)
    print("Saved summary to 'graph_summary.csv'.")
    print(padded_graphs[list(padded_graphs.keys())[0]][0])

    return padded_graphs

if __name__ == "__main__":
    main()