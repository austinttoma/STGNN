import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
import numpy as np
from temporal_gap_processor import normalize_time_gaps

class TemporalDataLoader:
    """
    DataLoader for temporal sequences of brain graphs.
    Groups graphs by subject and creates padded sequences for LSTM processing.
    """
    def __init__(self, dataset, subject_indices, encoder, device, batch_size=8, shuffle=True, seed=42,
                 exclude_target_visit=True, time_normalization='log', single_visit_horizon=6):
        self.dataset = dataset
        self.subject_indices = subject_indices
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.exclude_target_visit = exclude_target_visit
        self.time_normalization = time_normalization
        self.single_visit_horizon = single_visit_horizon
        
        # Group indices by subject
        self.subject_data = self._group_by_subject()
        self.subjects = list(self.subject_data.keys())
        
        if self.shuffle:
            # Use seeded randomness for reproducible shuffling
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.subjects)
    
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
            # Use seeded randomness for reproducible shuffling each epoch
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.subjects)
        
        for i in range(0, len(self.subjects), self.batch_size):
            batch_subjects = self.subjects[i:i + self.batch_size]
            
            # Collect all data objects and track their subject/visit mapping
            all_data = []
            subject_lengths = []
            subject_labels = []
            subject_time_gaps = []  # Time to predict for each subject
            visit_to_emb_idx = []  # List of (subject_idx, visit_idx) for each embedding
            
            processed_subj_idx = 0  # Track successfully processed subjects
            
            for subj_idx, subj_id in enumerate(batch_subjects):
                subject_indices = self.subject_data[subj_id]
                    
                visits = []
                labels = []
                time_gaps = []
                
                # Determine which visits to use for input vs target
                if self.exclude_target_visit:
                    if len(subject_indices) > 1:
                        # Multiple visits: Use all but last visit as input, last as target
                        input_indices = subject_indices[:-1]
                        target_idx = subject_indices[-1]
                        
                        # Get time to predict (from last input to target)
                        last_input_data = self.dataset[input_indices[-1]]
                        target_data = self.dataset[target_idx]
                        
                        # Calculate time gap to predict
                        if hasattr(last_input_data, 'visit_months') and hasattr(target_data, 'visit_months'):
                            # Convert tensors to scalars for calculation
                            last_months = last_input_data.visit_months.item() if hasattr(last_input_data.visit_months, 'item') else last_input_data.visit_months
                            target_months = target_data.visit_months.item() if hasattr(target_data.visit_months, 'item') else target_data.visit_months
                            time_to_predict = target_months - last_months
                        elif hasattr(last_input_data, 'months_to_next'):
                            months_to_next = last_input_data.months_to_next.item() if hasattr(last_input_data.months_to_next, 'item') else last_input_data.months_to_next
                            time_to_predict = months_to_next if not np.isnan(months_to_next) and months_to_next > 0 else 6
                        else:
                            time_to_predict = 6  # Default to 6 months if no temporal info
                        
                        # Use target's label
                        subject_labels.append(target_data.y.item())
                    else:
                        # Single visit: Use it as input, predict future (no explicit target)
                        input_indices = subject_indices
                        single_visit_data = self.dataset[input_indices[0]]
                        
                        # For single visits, predict a default time horizon ahead
                        # Use months_to_next if available, otherwise use configured default
                        if hasattr(single_visit_data, 'months_to_next'):
                            months_to_next = single_visit_data.months_to_next.item() if hasattr(single_visit_data.months_to_next, 'item') else single_visit_data.months_to_next
                            time_to_predict = months_to_next if not np.isnan(months_to_next) and months_to_next > 0 else self.single_visit_horizon
                        else:
                            time_to_predict = self.single_visit_horizon  # Configurable default prediction horizon
                        
                        # For single visits, we use the same label (assuming stable if no progression info)
                        # This is a clinical assumption: single visit = current state prediction
                        subject_labels.append(single_visit_data.y.item())
                    
                    # Normalize the time gap
                    time_to_predict_normalized = normalize_time_gaps(
                        np.array([time_to_predict]), 
                        method=self.time_normalization
                    )[0]
                    
                    subject_time_gaps.append(time_to_predict_normalized)
                    subject_lengths.append(len(input_indices))
                    
                    # Add input visits for embedding
                    for visit_idx, idx in enumerate(input_indices):
                        data = self.dataset[idx]
                        all_data.append(data)
                        visit_to_emb_idx.append((processed_subj_idx, visit_idx))  # Use processed index
                        labels.append(data.y.item())
                    
                    processed_subj_idx += 1  # Increment only for successfully processed subjects
                else:
                    # Use all visits and calculate meaningful time gaps
                    subject_lengths.append(len(subject_indices))
                    
                    # Calculate average time gap between consecutive visits for this subject
                    if len(subject_indices) > 1:
                        total_time_gap = 0
                        gap_count = 0
                        for i in range(len(subject_indices) - 1):
                            curr_data = self.dataset[subject_indices[i]]
                            next_data = self.dataset[subject_indices[i + 1]]
                            if hasattr(curr_data, 'visit_months') and hasattr(next_data, 'visit_months'):
                                curr_months = curr_data.visit_months.item() if hasattr(curr_data.visit_months, 'item') else curr_data.visit_months
                                next_months = next_data.visit_months.item() if hasattr(next_data.visit_months, 'item') else next_data.visit_months
                                total_time_gap += (next_months - curr_months)
                                gap_count += 1
                        
                        # Use average gap if available, otherwise default
                        avg_gap = (total_time_gap / gap_count) if gap_count > 0 else 6.0
                    else:
                        avg_gap = 6.0  # Default for single visits
                    
                    # Normalize the average time gap
                    avg_gap_normalized = normalize_time_gaps(
                        np.array([avg_gap]), 
                        method=self.time_normalization
                    )[0]
                    
                    subject_time_gaps.append(avg_gap_normalized)
                    
                    for visit_idx, idx in enumerate(subject_indices):
                        data = self.dataset[idx]
                        all_data.append(data)
                        visit_to_emb_idx.append((processed_subj_idx, visit_idx))
                        labels.append(data.y.item())
                    
                    # Use last label as subject label
                    subject_labels.append(labels[-1] if labels else 0)
                    processed_subj_idx += 1  # Increment for successfully processed subjects
            
            if not all_data:
                continue
            
            # Create large batch of all graphs
            big_batch = Batch.from_data_list(all_data).to(self.device)
            
            # Prepare time features for encoder if using time-aware GNN
            # Time features only make sense when excluding target visit (predicting future)
            if self.exclude_target_visit and len(subject_time_gaps) > 0:
                # Create time features for each graph in the batch
                batch_time_features = []
                for subj_idx, _ in visit_to_emb_idx:
                    # Use the subject's time gap for each of its visits
                    batch_time_features.append(subject_time_gaps[subj_idx])
                time_features_tensor = torch.tensor(batch_time_features, dtype=torch.float32, device=self.device)
            else:
                time_features_tensor = None
            
            # Compute all embeddings in one forward pass
            self.encoder.eval()
            with torch.no_grad():
                all_embeddings = self.encoder(big_batch.x, big_batch.edge_index, big_batch.batch, time_features_tensor)
            
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
            
            # Convert time gaps to tensor
            time_gaps_tensor = torch.tensor(subject_time_gaps, dtype=torch.float32, device=self.device)
            
            yield {
                'graph_seq': graph_seq,
                'lengths': lengths, 
                'labels': labels,
                'time_gaps': time_gaps_tensor,
                'batch_size': len(batch_sequences)  # Use actual number of sequences (may be less due to filtering)
            }
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.subjects) + self.batch_size - 1) // self.batch_size