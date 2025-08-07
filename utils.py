from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def stratified_subject_split(subject_label_dict, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(list(subject_label_dict.items()), columns=['subject_id', 'label'])

    # First stratified split: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=seed
    )

    # Second stratified split: val (10%) vs test (20%) from temp
    val_size = int(0.10 * len(df))  # 10% of full dataset
    val_df, test_df = train_test_split(
        temp_df, test_size=(len(df) - len(train_df) - val_size),
        stratify=temp_df['label'], random_state=seed
    )

    return list(train_df['subject_id']), list(val_df['subject_id']), list(test_df['subject_id'])

def train_val_test_split(kfold = 5, fold = 0, dataset_size = 1089, seed=123):
    n_sub = dataset_size
    id = list(range(n_sub))

    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state=seed+1)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id

def summarize_patient_graph_dims(padded_graphs):
    summaries = []

    for subj_id, graphs in padded_graphs.items():
        if not graphs or not hasattr(graphs[0], 'x') or graphs[0].x is None:
            print(f"Skipping {subj_id} (missing x)")
            continue
        try:
            nodes = [g.x.size(0) for g in graphs if g.x is not None]
            feats = [g.x.size(1) for g in graphs if g.x is not None]
            edges = [g.edge_index.size(1) for g in graphs if g.edge_index is not None]
            label = int(graphs[0].y.item()) if hasattr(graphs[0], 'y') else -1

            summaries.append({
                'subject_id': subj_id,
                'num_graphs': len(graphs),
                'avg_nodes': int(np.mean(nodes)),
                'avg_features': int(np.mean(feats)),
                'avg_edges': int(np.mean(edges)),
                'label': label
            })
        except Exception as e:
            print(f"Error processing {subj_id}: {e}")

    df = pd.DataFrame(summaries)
    print(f"Summary created with {len(df)} rows.")
    return df