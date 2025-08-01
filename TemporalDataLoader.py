import torch
from torch.nn.utils.rnn import pad_sequence  
from torch_geometric.data import Batch       
import numpy as np          

class TemporalDataLoader:
    """
    DataLoader for temporal sequences of brain graphs.
    Groups graphs by subject and creates padded sequences for LSTM processing.
    """
    
    def __init__(self, dataset, subject_indices, encoder, device, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.subject_indices = subject_indices
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by subject
        self.subject_data = self._group_by_subject()
        self.subjects = list(self.subject_data.keys())
        
        if self.shuffle:
            np.random.shuffle(self.subjects)
    
    def _group_by_subject(self):
        """Group dataset indices by subject ID."""
        subject_groups = {}
        for idx in self.subject_indices:
            data = self.dataset[idx]
            sid = getattr(data, 'subj_id', None)
            if sid is None:
                continue
            base_id = sid.split('_run')[0] if '_run' in sid else sid
            subject_groups.setdefault(base_id, []).append(idx)
        return subject_groups
    
    def _compute_subject_embeddings(self, subject_indices):
        # This method is no longer needed with the new batching approach, but keeping it for reference
        pass

    def _pad_sequences(self, sequences, labels):
        """Pad sequences to same length and create batch tensors."""
        batch_size = len(sequences)
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        max_len = lengths.max().item()
        embed_dim = sequences[0].size(1)
        
        # Create padded tensor
        padded = torch.zeros(batch_size, max_len, embed_dim, device=self.device)
        for i, seq in enumerate(sequences):
            padded[i, :seq.size(0)] = seq
        
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        return padded, lengths.to(self.device), labels_tensor
    
    def __iter__(self):
        """Iterate over temporal sequences in batches with batched embedding computation."""
        if self.shuffle:
            np.random.shuffle(self.subjects)
        
        for i in range(0, len(self.subjects), self.batch_size):
            batch_subjects = self.subjects[i:i + self.batch_size]
            
            # Collect all data objects and track their subject/visit mapping
            all_data = []
            subject_lengths = []
            subject_labels = []
            visit_to_emb_idx = []  # List of (subject_idx, visit_idx) for each embedding
            
            for subj_idx, subj_id in enumerate(batch_subjects):
                subject_indices = self.subject_data[subj_id]
                subject_lengths.append(len(subject_indices))
                visits = []
                labels = []
                
                for visit_idx, idx in enumerate(subject_indices):
                    data = self.dataset[idx]
                    all_data.append(data)
                    visit_to_emb_idx.append((subj_idx, visit_idx))
                    labels.append(data.y.item())
                
                # Use last label as subject label
                subject_labels.append(labels[-1] if labels else 0)
            
            if not all_data:
                continue
            
            # Create large batch of all graphs
            big_batch = Batch.from_data_list(all_data).to(self.device)
            
            # Compute all embeddings in one forward pass
            self.encoder.eval()
            with torch.no_grad():
                all_embeddings = self.encoder(big_batch.x, big_batch.edge_index, big_batch.batch)
            
            # Reshape embeddings back into per-subject sequences
            batch_sequences = []
            embed_dim = all_embeddings.size(-1)
            for subj_idx in range(len(batch_subjects)):
                subj_embs = []
                for emb_idx, (s_idx, v_idx) in enumerate(visit_to_emb_idx):
                    if s_idx == subj_idx:
                        subj_embs.append(all_embeddings[emb_idx])
                if subj_embs:
                    batch_sequences.append(torch.stack(subj_embs))
            
            # Pad sequences and create batch
            graph_seq, lengths, labels = self._pad_sequences(batch_sequences, subject_labels)
            
            yield {
                'graph_seq': graph_seq,
                'lengths': lengths, 
                'labels': labels,
                'batch_size': len(batch_subjects)
            }
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.subjects) + self.batch_size - 1) // self.batch_size