import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data

class DFC_ADNIDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        threshold=0.2,
        label_csv='TADPOLE_Simplified.csv',
        var_name='dynamic_fc',
        transform=None,
        pre_transform=None
    ):
        self.threshold = threshold
        self.label_csv = label_csv
        self.var_name = var_name
        super().__init__(root, transform, pre_transform)
        
        # Load data and subject IDs
        loaded_data = torch.load(self.processed_paths[0], weights_only=False)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
            self.data, self.slices, self.subj_id_list = loaded_data
        else:
            # Backward compatibility
            self.data, self.slices = loaded_data
            self.subj_id_list = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_dfc.pt']

    def download(self):
        pass

    def process(self):
        base_path = os.path.join(self.root, 'DFC_Matrices')
        label_path = os.path.join(self.root, self.label_csv)
        label_dict = self.load_subject_labels(label_path)

        data_list = []
        subj_id_list = []  # Store subject IDs separately
        npz_files = [f for f in os.listdir(base_path) if f.endswith('_dynamic_fc_matrix.npz')]

        for filename in tqdm(sorted(npz_files), desc="Processing DFC Matrices"):
            file_path = os.path.join(base_path, filename)
            try:
                base_name = filename.replace('_dynamic_fc_matrix.npz', '').replace(' copy', '')
                subj_id = base_name  # e.g., sub-002S0413_run-01
                # Extract the subject ID in format matching TADPOLE_Simplified.csv (e.g., 002S0413)
                if 'sub-' in subj_id:
                    base_subj_id = subj_id.split('_run')[0].replace('sub-', '')
                else:
                    base_subj_id = subj_id.split('_run')[0]

                if base_subj_id not in label_dict:
                    print(f"⚠️ No label found for subject {base_subj_id}, defaulting to 0")
                label = torch.tensor([label_dict.get(base_subj_id, 0)], dtype=torch.long)

                with np.load(file_path) as data:
                    if self.var_name not in data.files:
                        print(f"❌ '{self.var_name}' missing in {file_path}, skipping.")
                        continue

                    dfc_array = data[self.var_name]  # Shape: (T, N, N)

                if dfc_array.ndim != 3:
                    print(f"❌ Unexpected shape in {filename}: {dfc_array.shape}")
                    continue

                time_len = dfc_array.shape[0]

                for t in range(time_len):
                    matrix = dfc_array[t]
                    matrix = (matrix + matrix.T) / 2  # Ensure symmetry

                    graph = self.fc_to_graph(matrix)
                    graph.y = label
                    graph.time_index = torch.tensor([t], dtype=torch.long)
                    graph.subj_id = subj_id  # Will not persist, but can be used during preprocessing

                    data_list.append(graph)
                    subj_id_list.append(subj_id)  # Store subject ID

            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

        data, slices = self.collate(data_list)
        # Save subject IDs along with the data
        torch.save((data, slices, subj_id_list), self.processed_paths[0])

    def fc_to_graph(self, matrix, node_features=None):
        A = matrix.copy()
        N = A.shape[0]

        # Upper triangle without self-loops
        i, j = np.triu_indices(N, k=1)
        weights = A[i, j]

        # Apply threshold
        mask = np.abs(weights) >= self.threshold
        i = i[mask]
        j = j[mask]
        weights = weights[mask]

        # Add both directions to make undirected graph
        edge_index = np.concatenate([np.stack([i, j]), np.stack([j, i])], axis=1)
        edge_attr = np.concatenate([weights, weights])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)

        # Node features: use identity if none provided
        x = torch.eye(N, dtype=torch.float) if node_features is None else torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get(self, idx):
        """Override get method to attach subject ID to each data object"""
        data = super().get(idx)
        if self.subj_id_list is not None and idx < len(self.subj_id_list):
            data.subj_id = self.subj_id_list[idx]
        return data
    
    def load_subject_labels(self, label_csv_path, label_col='Label_CS_Num'):
        df = pd.read_csv(label_csv_path)
        df['Subject'] = df['Subject'].str.replace('_', '', regex=False)
        label_dict = dict(zip(df['Subject'], df[label_col]))
        return label_dict
