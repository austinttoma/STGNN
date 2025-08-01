import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data

class DFC_ADNIDataset(InMemoryDataset):
    def __init__(self, root, threshold=0.2, label_csv='TADPOLE_Simplified.csv', var_name='arr_0', transform=None, pre_transform=None):
        self.threshold = threshold
        self.label_csv = label_csv
        self.var_name = var_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        base_path = os.path.join(self.root, 'DFC_Matrices')
        label_path = os.path.join(self.root, self.label_csv)
        label_dict = self.load_subject_labels(label_path)

        data_list = []
        npz_files = [f for f in os.listdir(base_path) if f.endswith('_dynamic_fc_matrix.npz')]

        for filename in tqdm(sorted(npz_files), desc="Processing DFC Matrices"):
            file_path = os.path.join(base_path, filename)
            try:
                base_name = filename.replace('_dynamic_fc_matrix.npz', '')
                subj_id = base_name  # e.g., sub-XXXX_run-XX

                data = np.load(file_path)
                if self.var_name not in data:
                    print(f"'{self.var_name}' missing in {file_path}, skipping.")
                    continue

                dfc_array = data[self.var_name]  # Shape: (T, N, N)

                if dfc_array.ndim != 3:
                    print(f"Unexpected shape in {filename}: {dfc_array.shape}")
                    continue

                time_len = dfc_array.shape[0]
                base_subj_id = subj_id.split('_run')[0].replace('sub-', '')
                label = torch.tensor([label_dict.get(base_subj_id, 0)], dtype=torch.long)

                for t in range(time_len):
                    matrix = dfc_array[t]
                    matrix = (matrix + matrix.T) / 2  # ensure symmetry

                    graph = self.fc_to_graph(matrix, subj_id=f"{subj_id}_t{t}")
                    graph.y = label
                    graph.time_index = t
                    graph.subj_id = subj_id

                    data_list.append(graph)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def fc_to_graph(self, matrix, node_features=None, subj_id=None):
        A = matrix.copy()
        N = A.shape[0]
        A[np.abs(A) < self.threshold] = 0  # Apply threshold
        edge_index = np.array(np.nonzero(A))
        edge_attr = A[edge_index[0], edge_index[1]]

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)

        x = torch.eye(N, dtype=torch.float) if node_features is None else torch.tensor(node_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if subj_id is not None:
            data.subj_id = subj_id
        return data

    def load_subject_labels(self, label_csv_path, label_col='Label_CS_Num'):
        df = pd.read_csv(label_csv_path)
        df['Subject'] = df['Subject'].str.replace('_', '', regex=False)
        return dict(zip(df['Subject'], df[label_col]))