# STGNN: Alzheimer's Disease Progression Prediction

Spatiotemporal Graph Neural Network for predicting cognitive stage conversion using rs-fMRI data from ADNI.

## Overview
This project predicts Alzheimer's disease progression by combining Graph Neural Networks for brain connectivity analysis with RNNs for temporal modeling across patient visits. Best performance achieved with GraphSAGE-LSTM architecture: 67.7% test accuracy, 63.9% balanced accuracy, and 68.4% AUC.

**Paper**: "The Effectiveness of using rs-fMRI for Prediction of Cognitive Impairment Stage Conversion"

## Requirements
- Python 3.8+, CUDA GPU recommended
- Key packages: torch, torch-geometric, scikit-learn, nilearn

```bash
pip install -r requirements.txt
```

## Data Setup

### Required Data Files
1. **FC Matrices**: Place in `data/FC_Matrices/`
   - Format: `sub-XXXXXX_run-XX_fc_matrix.npz`
   - Each .npz file should contain a key named `fc_matrix` with the connectivity matrix

2. **Patient Labels**: `data/TADPOLE_TEMPORAL.csv`
   - Required columns: `Subject`, `Visit`, `Label_CS_Num`, `Visit_Order`, `Months_From_Baseline`, `Months_To_Next_Original`
   - Labels: 0=Stable, 1=Converter
   - Can be generated from TADPOLE_Simplified.csv and TADPOLE_COMPLETE.csv using `setup_temporal_data.py`

3. **Output Directory**: `model/` 
   - Created automatically if not present
   - Stores trained models and pretrained encoders

### Data Preparation (if needed)
```bash
# Generate TADPOLE_TEMPORAL.csv from original TADPOLE data
python setup_temporal_data.py
```

## Usage

### Basic Training (with default best parameters)
```bash
# Run with optimized defaults (GraphSAGE-LSTM, TopK pooling, focal loss)
python main.py

# Run with time features (experimental, requires --exclude_target_visit)
python main.py --use_time_features --exclude_target_visit
```

### Custom Configuration
```bash
python main.py [options]

# Model options
--model_type {LSTM,GRU,RNN}     # Temporal model architecture (default: LSTM)
--max_visits INT                # Maximum visits per patient sequence (default: 10)

# Training parameters  
--n_epochs INT                  # Number of training epochs (default: 100)
--batch_size INT                # Batch size for sequences (default: 16)
--lr FLOAT                      # Learning rate (default: 0.001)
--num_folds INT                 # Number of CV folds (default: 5)

# GNN architecture
--layer_type {GCN,GAT,GraphSAGE} # GNN layer type (default: GraphSAGE)
--gnn_hidden_dim INT            # GNN hidden dimension (default: 256)
--gnn_num_layers INT            # Number of GNN layers (default: 2)
--gnn_activation {relu,elu,gelu} # GNN activation function (default: elu)
--use_topk_pooling              # Use TopK pooling instead of global pooling
--topk_ratio FLOAT              # TopK pooling ratio (default: 0.3)

# RNN architecture
--lstm_hidden_dim INT           # Hidden dimension for RNN models (default: 64)
--lstm_num_layers INT           # Number of RNN layers (default: 1)
--lstm_bidirectional BOOL       # Use bidirectional LSTM (default: True)

# Loss function & class imbalance
--focal_alpha FLOAT             # Focal loss alpha for minority class (default: 0.90)
--focal_gamma FLOAT             # Focal loss gamma parameter (default: 3.0)
--label_smoothing FLOAT         # Label smoothing factor (default: 0.05)
--minority_focus_epochs INT     # Epochs to focus on minority class (default: 20)

# Model saving & pretraining
--save_path STR                 # Path to save models (default: ./model/)
--pretrain_encoder              # Enable GNN self-supervised pretraining
--use_pretrained                # Load existing pretrained encoder
--freeze_encoder                # Freeze encoder during temporal training
```

## Key Files
- `main.py` - Training script with 5-fold cross-validation and model selection
- `model.py` - Graph Neural Network with configurable layers (GCN/GAT/GraphSAGE) and TopK pooling
- `TemporalPredictor.py` - LSTM-based temporal classifier for sequences
- `GRUPredictor.py` / `RNNPredictor.py` - Alternative temporal models
- `FC_ADNIDataset.py` - PyTorch Geometric dataset for functional connectivity graphs
- `TemporalDataLoader.py` - Custom DataLoader for variable-length temporal sequences
- `FocalLoss.py` - Focal loss implementation for class imbalance
- `temporal_gap_processor.py` - Utilities for temporal feature processing
- `conversion_analyzer.py` - Analysis tools for conversion-specific accuracy
- `data/` - Preprocessed fMRI connectivity matrices and patient labels

## Acknowledgments
This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.
