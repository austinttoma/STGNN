# STGNN: Alzheimer's Disease Progression Prediction

Spatiotemporal Graph Neural Network for predicting cognitive stage conversion using rs-fMRI data from ADNI.

## Overview
This project predicts Alzheimer's disease progression by combining Graph Convolutional Networks for brain connectivity analysis with RNNs for temporal modeling across patient visits. Achieves 55% balanced accuracy and 26% minority F1 score with LSTM at 3 visits.

**Paper**: "The Effectiveness of using rs-fMRI for Prediction of Cognitive Impairment Stage Conversion"

## Requirements
- Python 3.8+, CUDA GPU recommended
- Key packages: torch, torch-geometric, scikit-learn, nilearn

```bash
pip install -r requirements.txt
```

## Data Setup
- Place FC matrices in: `data/FC_Matrices/` (format: `sub-XXXXXX_run-XX_fc_matrix.npz`)
- Patient labels: `data/TADPOLE_Simplified.csv`
- Trained models saved to: `model/` directory

## Usage
```bash
python main.py [options]

# Model options
--model_type {LSTM,GRU,RNN}     # Temporal model architecture
--max_visits INT                # Maximum visits per patient sequence (default: 10)

# Training parameters  
--n_epochs INT                  # Number of training epochs (default: 100)
--batch_size INT                # Batch size for sequences (default: 16)
--lr FLOAT                      # Learning rate (default: 0.005)
--num_folds INT                 # Number of CV folds (default: 5)

# Model architecture
--lstm_hidden_dim INT           # Hidden dimension for RNN models (default: 128)
--lstm_num_layers INT           # Number of RNN layers (default: 1)
--lstm_bidirectional BOOL       # Use bidirectional LSTM (default: True)

# Loss function & class imbalance
--focal_alpha FLOAT             # Focal loss alpha for minority class (default: 0.9)
--focal_gamma FLOAT             # Focal loss gamma parameter (default: 3.0)
--label_smoothing FLOAT         # Label smoothing factor (default: 0.05)

# Model saving & pretraining
--save_path STR                 # Path to save models (default: ./model/)
--pretrain_encoder              # Enable GNN self-supervised pretraining
--use_pretrained                # Load existing pretrained encoder
--freeze_encoder                # Freeze encoder during temporal training
```

## Key Files
- `main.py` - Training script with 5-fold cross-validation and model selection
- `model.py` - Graph Convolutional Network (3-layer GCN with graph normalization)
- `*Predictor.py` - Temporal models: LSTM, GRU, RNN, and Transformer architectures
- `FC_ADNIDataset.py` - PyTorch Geometric dataset for functional connectivity graphs
- `TemporalDataLoader.py` - Custom DataLoader for variable-length temporal sequences
- `FocalLoss.py` - Focal loss implementation for class imbalance
- `data/` - Preprocessed fMRI connectivity matrices and patient labels

## Acknowledgments
This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.