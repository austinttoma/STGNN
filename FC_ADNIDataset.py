import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
# temporal_gap_processor no longer needed - temporal data pre-calculated in TADPOLE_TEMPORAL.csv

class FC_ADNIDataset(InMemoryDataset):
    def __init__(self, root, threshold=0.2, label_csv='TADPOLE_TEMPORAL.csv', var_name='arr_0', transform=None, pre_transform=None):
        self.threshold = threshold
        self.label_csv = label_csv
        self.var_name = var_name
        self.subject_graph_dict = None
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
        base_path = os.path.join(self.root, 'FC_Matrices')
        label_path = os.path.join(self.root, self.label_csv)
        
        # Load labels and visit information
        label_dict, visit_dict = self.load_subject_labels_and_visits(label_path)

        fc_graphs = self.load_fc_graphs(base_path)

        data_list = []
        for subj_id, graph in fc_graphs.items():
            base_id = subj_id.split('_run')[0]
            graph.y = torch.tensor([label_dict.get(base_id, 0)], dtype=torch.long)
            graph.subj_id = subj_id
            
            # Add visit information if available
            if subj_id in visit_dict:
                visit_info = visit_dict[subj_id]
                graph.visit_code = visit_info.get('visit_code', 'unknown')
                graph.visit_months = visit_info.get('visit_months', 0)
                graph.months_to_next = visit_info.get('months_to_next', -1)
            elif base_id in visit_dict:
                # Fallback to base subject information
                visit_info = visit_dict[base_id]
                graph.visit_code = visit_info.get('visit_code', 'unknown')
                graph.visit_months = visit_info.get('visit_months', 0)
                graph.months_to_next = visit_info.get('months_to_next', -1)
            else:
                graph.visit_code = 'unknown'
                graph.visit_months = 0
                graph.months_to_next = -1
                
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def fc_to_graph(self, matrix, node_features=None, subj_id=None):
        A = matrix.copy()
        N = A.shape[0]
        A[np.abs(A) < self.threshold] = 0
        edge_index = np.array(np.nonzero(A))
        edge_attr = A[edge_index[0], edge_index[1]]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
        if node_features is None:
            # Use normalized identity matrix to better balance with time features
            x = torch.eye(N, dtype=torch.float) * 1.0  # Keep reasonable scale for GNN processing
        else:
            x = torch.tensor(node_features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if subj_id is not None:
            data.subj_id = subj_id
        return data

    def load_fc_graphs(self, base_path):
        fc_graphs = {}
        # Get all .npz files directly from the Updated FC Matrices folder
        npz_files = [f for f in os.listdir(base_path) if f.endswith('_fc_matrix.npz')]

        for filename in tqdm(sorted(npz_files), desc="Loading FC Matrices"):
            file_path = os.path.join(base_path, filename)
            
            try:
                # Extract subject ID and run number from filename
                # Format: sub-XXXXXX_run-XX_fc_matrix.npz
                base_name = filename.replace('_fc_matrix.npz', '')
                parts = base_name.split('_run-')
                subj_id = parts[0].replace('sub-', '')
                run_num = parts[1] if len(parts) > 1 else '01'
                
                data = np.load(file_path)
                if self.var_name not in data:
                    print(f"'{self.var_name}' missing in {file_path}, skipping.")
                    continue

                fc_matrix = data[self.var_name]
                fc_matrix = (fc_matrix + fc_matrix.T) / 2  # Ensure symmetry

                full_id = f"{subj_id}_run{run_num}"
                graph = self.fc_to_graph(fc_matrix, subj_id=full_id)
                fc_graphs[full_id] = graph

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        return fc_graphs


    def load_subject_labels(self, label_csv_path, label_col='Label_CS_Num'):
        df = pd.read_csv(label_csv_path)
        df['Subject'] = df['Subject'].str.replace('_', '', regex=False)
        return dict(zip(df['Subject'], df[label_col]))
    
    def load_subject_labels_and_visits(self, label_csv_path, label_col='Label_CS_Num'):
        df = pd.read_csv(label_csv_path)
        df['Subject'] = df['Subject'].str.replace('_', '', regex=False)
        
        # Sort by subject and visit order to ensure proper chronological mapping
        df = df.sort_values(['Subject', 'Visit_Order'])
        
        # Create label dictionary use the label from the last visit per subject
        label_dict = {}
        for subject in df['Subject'].unique():
            subject_data = df[df['Subject'] == subject]
            # Use the last visit's label as the overall subject label
            label_dict[subject] = subject_data.iloc[-1][label_col]
        
        # Create visit info dictionary mapping subject_runXX to specific visit data
        visit_dict = {}
        
        for subject in df['Subject'].unique():
            subject_data = df[df['Subject'] == subject].sort_values('Visit_Order')
            
            # Map each run number to chronologically ordered visits
            for run_idx, (_, visit_row) in enumerate(subject_data.iterrows()):
                run_key = f"{subject}_run{run_idx + 1:02d}"  # Format as run01, run02, etc.
                
                # Store visit information for this specific run
                visit_dict[run_key] = {
                    'visit_code': visit_row['Visit'],
                    'visit_months': visit_row['Months_From_Baseline'],
                    'months_to_next': visit_row.get('Months_To_Next_Original', -1)
                }
                
                # Also store base subject info (for backward compatibility)
                if run_idx == 0:  # First visit
                    visit_dict[subject] = visit_dict[run_key].copy()
        
        return label_dict, visit_dict
