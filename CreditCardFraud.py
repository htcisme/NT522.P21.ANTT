

# --- Imports ---
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import random
import copy
import os
import time
from collections import deque, namedtuple, Counter
import traceback # For detailed error printing
import matplotlib.pyplot as plt
import seaborn as sns
import json

from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv # Changed from GATConv
import torch_geometric.utils

from imblearn.over_sampling import RandomOverSampler # For oversampling

from sklearn.preprocessing import StandardScaler # Or MinMaxScaler as per paper's description
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, accuracy_score, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
config = {
    # --- General ---
    'seed': 42,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # --- Data ---
    'csv_file_path': "/kaggle/input/creditcardfraud/creditcard.csv", # <-- **UPDATE THIS PATH**
    'time_col': 'Time',
    'amount_col': 'Amount',
    'label_col': 'Class',
    'test_size': 0.15, # Note: FL setup currently uses all client data, no explicit test set during FL rounds
    'val_size': 0.15,  # Note: FL setup currently uses all client data, no explicit val set during FL rounds
    'perform_oversampling': True, # Flag to enable/disable oversampling
    'oversampling_target_fraud_ratio': 0.02, # Target fraud ratio (2%)

    # --- Graph Building ---
    'graph_max_neighbors': 30,       # Lookback window size (Keep or Tune)
    'graph_time_window': 60 * 60,  # Time window (1 hour in seconds - Align with paper)
    'graph_use_feature_similarity': True,
    'graph_similarity_threshold': 0.9, # Cosine similarity threshold (Align with paper)
    'graph_verbose': True,             # Print progress during graph building

    # --- GNN Model (Modified SimpleTSSGCNet with GCN) ---
    'gnn_hidden_dim': 64,         # Align with paper
    'gnn_dropout': 0.5,           # Keep or Tune
    # 'gat_heads' removed as we switched to GCN
    # feature_cols will be set after defining V_cols
    # input_dim will be determined from feature_cols

    # --- Reinforcement Learning ---
    'use_feature_weighting': True, # Keep True to align with RL action space in paper
    'threshold_min': 0.05,
    'threshold_max': 0.95,
    'threshold_num_actions': 19,   # Number of discrete thresholds
    'rl_state_dim': None,          # Will be calculated based on GNN output (GCN based)
    'rl_lr': 1e-4,                 # Align with paper or original code (keep 1e-4 for now)
    'rl_buffer_capacity': 50000,
    'rl_batch_size': 64,           # Align with paper
    'rl_gamma': 0.99,              # Discount factor
    'rl_epsilon_start': 1.0,
    'rl_epsilon_end': 0.05,
    'rl_epsilon_decay': 0.998,
    'rl_target_update_freq': 10,
    'lambda_fpr': 0.3,             # Weight for FPR in reward - **TUNE THIS**
    'rl_updates_per_round': 2,

    # --- Federated Learning ---
    'num_clients': 10,
    'fl_rounds': 100,              # Increase for real training (Start lower for testing)
    'fl_clients_per_round': 3,
    'fl_local_epochs': 1,
    'gnn_lr': 1e-3,               # GNN learning rate for local updates (Align with paper)
    'fl_early_stopping_patience': 15, # Keep or Tune

    # --- Evaluation ---
    'recall_k_percent': 0.05, # Recall@1% as used in paper's Table 2
    
        # --- Web App ---
    'model_save_path': './models/',
    'checkpoint_save_path': '/kaggle/input/checkpoints/pytorch/default/2/',
    'checkpoint_save' : './checkpoinst/',
    'web_port': 5000,
}

# Define V columns and feature columns used by the model
v_cols = [f'V{i}' for i in range(1, 29)]
config['feature_cols'] = [config['time_col'], config['amount_col']] + v_cols
config['num_features'] = len(config['feature_cols'])

# Calculate RL state dimension based on **GCN** architecture
# Output from GCN + Output from GRU (Temporal)
config['rl_state_dim'] = config['gnn_hidden_dim'] + config['gnn_hidden_dim']
def save_checkpoint(gnn_model, rl_agent, target_rl_agent, optimizer_rl, replay_buffer, 
                   stopper, current_round, epsilon, config, filename="checkpoint.pth"):
    """Save training state to checkpoint file."""
    os.makedirs(config['checkpoint_save'], exist_ok=True)
    checkpoint_path = os.path.join(config['checkpoint_save'], filename)
    
    state = {
        'round': current_round + 1,
        'gnn_model_state_dict': gnn_model.state_dict(),
        'rl_agent_state_dict': rl_agent.state_dict(),
        'target_rl_agent_state_dict': target_rl_agent.state_dict(),
        'optimizer_rl_state_dict': optimizer_rl.state_dict(),
        'replay_buffer': list(replay_buffer.memory),
        'epsilon': epsilon,
        'early_stopper_best_score': stopper.best_score,
        'early_stopper_counter': stopper.counter,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved successfully at round {current_round} to {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint at round {current_round}: {e}")
        return False

def load_checkpoint(filename="checkpoint (2).pth"):
    """Load training state from checkpoint file."""
    checkpoint_path = os.path.join(config['checkpoint_save_path'], filename)
    
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=config['device'])
            
            # Convert replay buffer from list back to deque
            if 'replay_buffer' in checkpoint and isinstance(checkpoint['replay_buffer'], list):
                buffer_capacity = checkpoint.get('config', {}).get('rl_buffer_capacity', 50000)
                checkpoint['replay_buffer'] = deque(checkpoint['replay_buffer'], maxlen=buffer_capacity)
            
            print("Checkpoint loaded successfully.")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return None
    else:
        print(f"Checkpoint file not found at {checkpoint_path}.")
        return None

def save_final_models(gnn_model, rl_agent, config, metrics_history=None):
    """Save final trained models."""
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # Save GNN model
    gnn_path = os.path.join(config['model_save_path'], "final_gnn_model.pth")
    torch.save(gnn_model.state_dict(), gnn_path)
    
    # Save RL agent
    rl_path = os.path.join(config['model_save_path'], "final_rl_agent.pth")
    torch.save(rl_agent.state_dict(), rl_path)
    
    # Save model metadata
    metadata = {
        'gnn_config': {
            'input_dim': config['num_features'],
            'hidden_dim': config['gnn_hidden_dim'],
            'num_classes': 2,
            'dropout_rate': config['gnn_dropout']
        },
        'rl_config': {
            'state_dim': config['rl_state_dim'],
            'num_actions': config['threshold_num_actions'],
            'num_features': config['num_features']
        },
        'training_config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    if metrics_history:
        metadata['final_metrics'] = {
            'final_f1': metrics_history['f1'][-1] if metrics_history['f1'] else 0,
            'final_auc_pr': metrics_history['auc_pr'][-1] if metrics_history['auc_pr'] else 0,
            'final_recall_k': metrics_history['recall_k'][-1] if metrics_history['recall_k'] else 0
        }
    
    metadata_path = os.path.join(config['model_save_path'], "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Models saved to: {config['model_save_path']}")
    return gnn_path, rl_path, metadata_path


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed}")

def recall_at_k_percent(y_true, scores, k_percent=0.01): # Default to 1%
    """Calculates recall for the top k% highest scores."""
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(scores, np.ndarray): scores = np.array(scores)

    if np.sum(y_true) == 0: return 0.0 # No positive samples
    num_samples = len(scores)
    k = int(np.ceil(num_samples * k_percent)) # Use ceil to ensure at least 1 if k_percent is small
    if k <= 0: k = 1 # Ensure k is at least 1

    try:
        # Handle potential NaNs in scores
        nan_mask = np.isnan(scores)
        if np.all(nan_mask): return 0.0 # All scores are NaN
        valid_scores = scores[~nan_mask]
        valid_y_true = y_true[~nan_mask]
        if len(valid_scores) == 0: return 0.0

        # Sort based on valid scores
        indices_sorted_by_score = np.argsort(valid_scores)[::-1]
        top_k_indices = indices_sorted_by_score[:k]
        top_k_true_labels = valid_y_true[top_k_indices]

        # Calculate recall based on valid positive samples
        total_positives = np.sum(valid_y_true)
        if total_positives == 0: return 0.0
        recall = np.sum(top_k_true_labels) / total_positives
        return recall
    except Exception as e:
        print(f"Error in recall_at_k_percent (k={k}, k%={k_percent}): {e}. Returning 0.0")
        # print(f"y_true shape: {y_true.shape}, scores shape: {scores.shape}, k: {k}")
        # print(f"y_true unique: {np.unique(y_true)}, scores has NaN: {np.isnan(scores).any()}")
        return 0.0


def calculate_reward(y_true, y_pred, y_scores, lambda_fpr=0.5):
    """Calculates reward based on F1 score and False Positive Rate (FPR)."""
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = 0, 0, 0, 0
    fpr = 0.0

    try:
        # Use labels=[0, 1] to ensure matrix shape is consistent
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate FPR only if there are actual negatives
        actual_negatives = tn + fp
        if actual_negatives > 0:
            fpr = fp / actual_negatives
        else:
            fpr = 0.0 # Or handle as appropriate if no negatives exist

    except ValueError as e:
        # This might happen if y_true contains only one class
        print(f"Warning: ValueError calculating confusion matrix or FPR: {e}. Check label distribution.")
        # Fallback FPR calculation (less robust)
        num_negatives = np.sum(y_true == 0)
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        fpr = false_positives / num_negatives if num_negatives > 0 else 0.0
    except Exception as e_gen:
        print(f"Warning: Unexpected error calculating FPR: {e_gen}. Setting FPR to 0.")
        fpr = 0.0

    reward = f1 - lambda_fpr * fpr
    # Ensure reward is not NaN
    reward = np.nan_to_num(reward)
    return reward

# --- Graph Building Function (Minor adjustments possible, core logic kept) ---
def build_graph_from_cc_df(df, feature_cols, time_col='Time', label_col='Class',
                           max_neighbors=20, time_window=3600,
                           use_feature_similarity=True, similarity_threshold=0.9,
                           verbose=True):
    """Builds a PyTorch Geometric graph from a Credit Card DataFrame."""
    df = df.reset_index(drop=True) # Ensure indices are 0, 1, 2,...
    start_time_build = time.time()
    if verbose: print(f"Starting graph build for {len(df)} nodes...")

    # --- Input Validation ---
    if time_col not in df.columns: raise ValueError(f"Missing time column: {time_col}")
    if label_col not in df.columns: raise ValueError(f"Missing label column: {label_col}")
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features: raise ValueError(f"Missing feature columns: {missing_features}")
    if df.empty:
        print("Warning: Input DataFrame is empty. Returning empty graph.")
        return Data(x=torch.empty((0, len(feature_cols)), dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    y=torch.empty((0,), dtype=torch.long),
                    timestamp=torch.empty((0,), dtype=torch.float),
                    num_nodes=0)

    # --- Feature and Label Extraction ---
    try:
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        timestamps = torch.tensor(df[time_col].values, dtype=torch.float)
        y = torch.tensor(df[label_col].values, dtype=torch.long)
    except Exception as e:
        raise ValueError(f"Error converting DataFrame columns to tensors: {e}")

    # --- Feature Normalization for Similarity (if enabled) ---
    x_normalized_for_sim = None
    if use_feature_similarity:
        if verbose: print("Normalizing features for similarity calculation...")
        # Use V1-V28 + Amount for similarity as mentioned in paper/common practice
        sim_feature_cols = [f'V{i}' for i in range(1, 29)] + [config['amount_col']]
        present_sim_cols = [col for col in sim_feature_cols if col in df.columns]
        if present_sim_cols:
            try:
                features_to_normalize = df[present_sim_cols].values
                # Handle potential NaNs before scaling
                features_to_normalize = np.nan_to_num(features_to_normalize)
                if features_to_normalize.shape[0] > 0: # Only scale if data exists
                    scaler_sim = StandardScaler()
                    x_normalized_for_sim = scaler_sim.fit_transform(features_to_normalize)
                    if verbose: print("Normalization for similarity complete.")
                else:
                    print("Warning: No data to normalize for similarity.")
                    use_feature_similarity = False # Disable if no data
            except Exception as e:
                print(f"Warning: Error during feature normalization for similarity: {e}. Disabling feature similarity.")
                use_feature_similarity = False
        else:
            print("Warning: No V1-V28/Amount cols found for similarity calculation. Disabling feature similarity.")
            use_feature_similarity = False

    # --- Edge Building ---
    if verbose: print(f"Building edges (max_neighbors={max_neighbors}, time_window={time_window}s, use_sim={use_feature_similarity}, sim_thresh={similarity_threshold})...")
    edge_index_list = []
    n_nodes = len(df)
    num_edges_added = 0
    edge_build_start_time = time.time()

    for i in range(n_nodes):
        if verbose and i > 0 and (i % (n_nodes // 10 or 1) == 0 or i == n_nodes - 1):
             current_time = time.time()
             elapsed = current_time - edge_build_start_time
             rate = (i + 1) / elapsed if elapsed > 0 else 0
             est_total = n_nodes / rate if rate > 0 else 0
             print(f"  Processed node {i+1}/{n_nodes} ({num_edges_added} edges). Rate: {rate:.1f} nodes/s. Est. total: {est_total:.1f}s")

        t_i = timestamps[i]
        # Iterate backwards through potential neighbors within the lookback window
        for k in range(1, max_neighbors + 1):
            j = i - k
            if j < 0: break # Reached beginning of dataframe

            t_j = timestamps[j]
            # Check time window constraint first
            if (t_i - t_j) > time_window:
                break # Neighbors are too old, stop searching for this node i

            # If within time window, check similarity (if enabled)
            is_neighbor = False
            if use_feature_similarity and x_normalized_for_sim is not None:
                # Ensure indices are valid
                if i < x_normalized_for_sim.shape[0] and j < x_normalized_for_sim.shape[0]:
                    try:
                        # Calculate cosine similarity safely
                        vec_i = x_normalized_for_sim[i].reshape(1, -1)
                        vec_j = x_normalized_for_sim[j].reshape(1, -1)
                        sim = cosine_similarity(vec_i, vec_j)[0, 0]
                        sim = np.clip(sim, -1.0, 1.0) # Clip for numerical stability
                        if sim >= similarity_threshold:
                            is_neighbor = True
                    except Exception as e_sim:
                        # print(f"Warning: Error calculating similarity between {i} and {j}: {e_sim}")
                        pass # Skip this potential neighbor pair if error occurs
                else:
                     # This case should ideally not happen if df is indexed correctly
                     # print(f"Warning: Index out of bounds during similarity check ({i}, {j} vs {x_normalized_for_sim.shape[0]})")
                     pass
            elif not use_feature_similarity:
                 is_neighbor = True # Connect if within time window and max_neighbors

            if is_neighbor:
                # Add edge in both directions
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])
                num_edges_added += 2

    if verbose: print(f"Edge building loop finished. Total potential bidirectional edges added: {num_edges_added}")

    # --- Final Graph Creation ---
    if not edge_index_list:
         print(f"WARNING: No edges created for graph (size {n_nodes}). Check parameters (time_window, max_neighbors, similarity_threshold).")
         edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        # Remove self-loops and duplicates (coalesce)
        edge_index_tensor, _ = torch_geometric.utils.remove_self_loops(edge_index_tensor)
        edge_index_tensor = torch_geometric.utils.coalesce(edge_index_tensor) # Sorts and removes duplicates

    data = Data(x=x, edge_index=edge_index_tensor, y=y, timestamp=timestamps, num_nodes=n_nodes)
    end_time_build = time.time()
    if verbose: print(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges (final, unique, no self-loops). Build time: {end_time_build - start_time_build:.2f}s")
    return data


# --- GNN Model Definition (Using GCN based on paper's justification) ---
class SimpleTSSGCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Spatial Modeling (using GCN)
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.gcn3 = GCNConv(hidden_dim, hidden_dim) # Optional 3rd layer
        # self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Temporal Modeling (using GRU as per paper/original code)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.beta = nn.Parameter(torch.tensor(0.1)) # Learnable decay parameter for time attention

        # Output Layer
        # Concatenating output of GCN (spatial) and GRU (temporal)
        final_concat_dim = hidden_dim + hidden_dim
        self.out_linear = nn.Linear(final_concat_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Note: Semantic Modeling component is omitted as credit card data lacks explicit type features.

    def temporal_attention(self, edge_index, x, timestamps, current_node_indices):
        """Applies time-aware attention and GRU for temporal modeling."""
        device = x.device
        timestamps = timestamps.float().to(device)
        batch_size = len(current_node_indices)
        # Initialize output tensor for the batch
        temp_outputs = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Efficiently find neighbors for all nodes in the batch
        # Ensure edge_index is on the correct device
        edge_index = edge_index.to(device)

        # Create a mask for edges where the destination node is in the current batch
        # Need mapping from global node index to batch index if current_node_indices is not dense
        is_dense_indices = torch.all(current_node_indices == torch.arange(batch_size, device=device))
        if not is_dense_indices:
            node_idx_to_batch_idx = {node_idx.item(): i for i, node_idx in enumerate(current_node_indices)}
        else:
            node_idx_to_batch_idx = None # Direct mapping

        # Find edges pointing TO nodes in the current batch
        relevant_edges_mask = torch.isin(edge_index[1], current_node_indices)
        relevant_edge_index = edge_index[:, relevant_edges_mask] # Edges (source, dest) where dest is in batch

        # Iterate through each node in the *batch*
        for i, node_idx_tensor in enumerate(current_node_indices):
            node_idx = node_idx_tensor.item()
            batch_idx = i # Index within the current batch/temp_outputs tensor

            # Find neighbors (source nodes) for this specific destination node_idx
            mask = (relevant_edge_index[1] == node_idx)
            neighbor_global_indices = relevant_edge_index[0][mask]

            if len(neighbor_global_indices) == 0:
                continue # No neighbors found for this node in the graph

            # Ensure neighbor indices are valid within the full 'x' tensor
            valid_neighbor_mask = neighbor_global_indices < x.size(0)
            valid_neighbor_indices = neighbor_global_indices[valid_neighbor_mask]

            if len(valid_neighbor_indices) == 0:
                continue # No *valid* neighbors found

            # Get features and timestamps of valid neighbors
            neighbor_feats = x[valid_neighbor_indices]
            neighbor_times = timestamps[valid_neighbor_indices]

            # Current node's timestamp
            # Handle potential index out of bounds for timestamp if node_idx is too large
            if node_idx >= len(timestamps):
                 # print(f"Warning: node_idx {node_idx} out of bounds for timestamps (len {len(timestamps)}). Skipping temporal attention.")
                 continue
            now_time = timestamps[node_idx]

            # Calculate time difference and attention weights
            delta_t = F.relu(now_time - neighbor_times) # Time difference should be non-negative
            alpha = torch.exp(-self.beta * delta_t) # Time-aware attention weights

            # Normalize attention weights
            alpha_sum = alpha.sum() + 1e-9 # Add epsilon for numerical stability
            alpha_norm = alpha / alpha_sum

            # Weight neighbor features by attention
            # Unsqueeze alpha_norm to match feature dimensions for broadcasting
            feats_weighted = neighbor_feats * alpha_norm.unsqueeze(-1)

            # Apply GRU to the sequence of weighted neighbor features
            if feats_weighted.shape[0] > 0:
                # GRU expects input shape (batch, seq_len, input_size)
                # Here, batch=1, seq_len=num_neighbors, input_size=feature_dim
                gru_input = feats_weighted.unsqueeze(0)
                # Initialize hidden state for GRU
                h0 = torch.zeros(1, 1, self.hidden_dim, device=device) # (num_layers * num_directions, batch, hidden_size)
                gru_out, _ = self.gru(gru_input, h0)
                # Use the output of the last time step
                temp_outputs[batch_idx] = gru_out[0, -1] # gru_out shape (batch, seq_len, hidden_size)

        return temp_outputs

    def forward(self, data):
        x, edge_index, timestamp = data.x, data.edge_index, data.timestamp
        # Ensure timestamp tensor exists and move to device
        if timestamp is None: raise ValueError("Timestamp tensor is missing in data object")
        timestamp = timestamp.to(x.device)
        # Ensure edge_index exists (can be empty, but not None)
        if edge_index is None:
            print("Warning: edge_index is None. Creating empty tensor.")
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)


        # --- Spatial Path (GCN) ---
        # Check if edges exist for GCN layers
        if edge_index.shape[1] > 0:
            h1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
            # h1 = self.dropout(h1) # Dropout after activation/norm often better
            h2 = F.relu(self.bn2(self.gcn2(h1, edge_index)))
            # h2 = self.dropout(h2)
            # spatial_output = F.relu(self.bn3(self.gcn3(h2, edge_index))) # If using 3 layers
            spatial_output = h2 # Using 2 layers
        else:
            # Handle case with no edges: spatial features are zero
            print("Warning: No edges found in graph during forward pass. Spatial output will be zeros.")
            spatial_output = torch.zeros(x.size(0), self.hidden_dim, device=x.device)


        # --- Temporal Path (GRU) ---
        # Apply temporal attention to all nodes in the current graph/batch
        node_indices = torch.arange(x.size(0), device=x.device)
        temporal_output = self.temporal_attention(edge_index, x, timestamp, node_indices)


        # --- Fusion ---
        fused = torch.cat([spatial_output, temporal_output], dim=1)
        fused = self.dropout(fused) # Apply dropout before final layer


        # --- Output ---
        out = self.out_linear(fused)
        # No softmax here, CrossEntropyLoss will apply it

        return out

    def get_graph_embedding(self, data):
        """Generates a graph-level embedding (e.g., for RL state)."""
        x, edge_index, timestamp = data.x, data.edge_index, data.timestamp
        if timestamp is None: raise ValueError("Timestamp tensor is missing")
        timestamp = timestamp.to(x.device)
        if edge_index is None:
             edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)


        with torch.no_grad(): # No need to track gradients for embedding generation
            # Spatial Path (GCN)
            if edge_index.shape[1] > 0:
                h1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
                h2 = F.relu(self.bn2(self.gcn2(h1, edge_index)))
                # spatial_output = F.relu(self.bn3(self.gcn3(h2, edge_index))) # If using 3 layers
                spatial_output = h2 # Using 2 layers
            else:
                spatial_output = torch.zeros(x.size(0), self.hidden_dim, device=x.device)


            # Temporal Path (GRU)
            node_indices = torch.arange(x.size(0), device=x.device)
            temporal_output = self.temporal_attention(edge_index, x, timestamp, node_indices)


            # Fusion
            fused_node_embeddings = torch.cat([spatial_output, temporal_output], dim=1)


            # Aggregation (Mean pooling for graph embedding)
            if fused_node_embeddings.size(0) > 0:
                graph_embedding = fused_node_embeddings.mean(dim=0)
            else:
                # Handle empty graph case
                graph_embedding = torch.zeros(self.hidden_dim + self.hidden_dim, device=x.device)

        return graph_embedding.detach() # Detach from computation graph

# --- RL Components ---
Transition = namedtuple('Transition', ('state', 'action_index', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        actual_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent(nn.Module):
    """DQN Agent with separate heads for Q-values and Feature Weights."""
    def __init__(self, state_dim, num_actions, num_features):
        super().__init__()
        self.num_features = num_features
        # Shared layers can be adjusted
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            # nn.Linear(128, 128) # Optional second shared layer
        )
        # Q-value head (outputs value for each discrete action/threshold)
        self.q_value_head = nn.Sequential(
            nn.Linear(128, 128), # Layer before output
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        # Feature weighting head (outputs a weight for each input feature)
        self.feature_weight_head = nn.Sequential(
            nn.Linear(128, 128), # Layer before output
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Sigmoid() # Output weights between 0 and 1
        )

    def forward(self, state):
        shared_output = self.shared_net(state)
        q_values = self.q_value_head(shared_output)
        feature_weights = self.feature_weight_head(shared_output)
        return q_values, feature_weights

def select_action_epsilon_greedy(state, dqn_agent, num_actions, epsilon, device):
    """Selects an action using epsilon-greedy strategy."""
    if random.random() > epsilon:
        # Exploit: choose the best action based on Q-values
        with torch.no_grad():
            # Ensure state is correctly shaped (batch dimension) and on device
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(device)
            # Get Q-values (ignore feature weights for action selection)
            q_values, _ = dqn_agent(state)
            # Choose action with the highest Q-value
            action_index = q_values.max(1)[1].item() # .item() gets the Python number
    else:
        # Explore: choose a random action
        action_index = random.randrange(num_actions)
    return action_index

def train_rl_agent(dqn_agent, target_dqn, optimizer_rl, replay_buffer, batch_size, gamma, device):
    """Performs one step of optimization for the DQN agent."""
    if len(replay_buffer) < batch_size:
        return None # Not enough samples in buffer yet

    transitions = replay_buffer.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details)
    batch = Transition(*zip(*transitions))

    # Filter out transitions where the state might be None (should not happen if pushed correctly)
    valid_indices = [i for i, s in enumerate(batch.state) if s is not None]
    if not valid_indices: return None # No valid states in the batch

    # Create batches of states, actions, rewards, etc. for valid transitions
    state_batch = torch.stack([batch.state[i].to(device) for i in valid_indices]).to(device)
    action_batch = torch.tensor([batch.action_index[i] for i in valid_indices], dtype=torch.long, device=device).unsqueeze(1) # Shape [batch_size, 1]
    reward_batch = torch.tensor([batch.reward[i] for i in valid_indices], dtype=torch.float32, device=device)
    done_batch = torch.tensor([batch.done[i] for i in valid_indices], dtype=torch.bool, device=device)

    # Handle next states (some might be None if it was a terminal transition)
    next_state_list = [batch.next_state[i] for i in valid_indices]
    non_final_mask = torch.tensor([s is not None for s in next_state_list], dtype=torch.bool, device=device)

    # Stack only the non-final next states
    non_final_next_states_list = [s.to(device) for s in next_state_list if s is not None]
    if non_final_next_states_list: # Check if the list is not empty
        non_final_next_states = torch.stack(non_final_next_states_list)
    else:
        # Create an empty tensor with the correct shape if no non-final next states exist
        non_final_next_states = torch.empty((0, state_batch.shape[1]), device=device) # state_dim = state_batch.shape[1]

    # --- Compute Q(s_t, a) ---
    # Get Q-values for all actions from the main DQN agent
    current_q_values_all, _ = dqn_agent(state_batch)
    # Select the Q-value for the action that was actually taken
    current_q_values_for_action = current_q_values_all.gather(1, action_batch)

    # --- Compute V(s_{t+1}) for non-final states ---
    next_q_values = torch.zeros(len(valid_indices), device=device) # Initialize with zeros
    # Use the target network to estimate the value of the next state
    if non_final_next_states.nelement() > 0: # Check if tensor is not empty
         with torch.no_grad():
            # Get Q-values for all actions in the next states from the target network
            next_q_values_all, _ = target_dqn(non_final_next_states)
            # Select the maximum Q-value for each next state (greedy policy)
            next_q_values[non_final_mask] = next_q_values_all.max(1)[0] # Use the mask to place values correctly

    # --- Compute the expected Q values (target) ---
    # target = reward + gamma * V(s_{t+1})
    # V(s_{t+1}) is 0 if the state was final (done=True)
    target_q_values = reward_batch + (gamma * next_q_values * (~done_batch)) # ~done_batch ensures target is just reward if done

    # --- Compute Loss (e.g., Smooth L1 or MSE) ---
    # loss = F.smooth_l1_loss(current_q_values_for_action, target_q_values.unsqueeze(1))
    loss = F.mse_loss(current_q_values_for_action, target_q_values.unsqueeze(1))

    # --- Optimize the model ---
    optimizer_rl.zero_grad()
    loss.backward()
    # Optional: Gradient clipping
    # torch.nn.utils.clip_grad_value_(dqn_agent.parameters(), 100)
    optimizer_rl.step()

    return loss.item()


# --- Federated Learning Components ---

class EarlyStopper:
    """Simple early stopping implementation."""
    def __init__(self, patience=10, min_delta=0.0001): # Added min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.early_stop = False

    def step(self, score):
        current_score = score if not (score is None or np.isnan(score)) else -float("inf")

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            # print(f"EarlyStopping: New best score: {self.best_score:.4f}") # Debug print
        else:
            self.counter += 1
            # print(f"EarlyStopping: Counter: {self.counter}/{self.patience}. Best: {self.best_score:.4f}, Current: {current_score:.4f}") # Debug print
            if self.counter >= self.patience:
                print(f"EarlyStopping: Stopping training. No improvement for {self.patience} rounds.")
                self.early_stop = True
        return self.early_stop

def local_update(data, model, local_epochs=1, lr=1e-3, device='cuda', label_col='y'):
    """Performs local training update on a client's data."""
    # Create a copy of the global model for local training
    local_model = copy.deepcopy(model).to(device)
    optimizer_gnn = optim.Adam(local_model.parameters(), lr=lr)
    local_model.train() # Set to training mode

    data = data.to(device)

    # --- Input data validation ---
    if data.x is None or data.x.shape[0] == 0:
        print(f"Warning: Skipping local update due to empty node features (x).")
        return model.cpu().state_dict() # Return original weights if no data
    if data.edge_index is None or data.edge_index.shape[1] == 0:
        # Allow training even with no edges, GNN layers should handle this
        print(f"Warning: Training locally with empty edge_index.")
    if data.y is None or data.y.shape[0] == 0:
        print(f"Warning: Skipping local update due to missing labels (y).")
        return model.cpu().state_dict() # Return original weights if no labels
    if data.y.shape[0] != data.x.shape[0]:
        print(f"Warning: Mismatch between number of nodes ({data.x.shape[0]}) and labels ({data.y.shape[0]}). Skipping update.")
        return model.cpu().state_dict()

    labels = getattr(data, label_col) # Get labels using attribute name

    # --- Class Weighting (for imbalanced data) ---
    weight = None
    if labels.dim() > 0 and labels.shape[0] > 0: # Calculate weights only if labels exist
        n_samples = len(labels)
        n_classes = 2 # Assuming binary classification
        counts = torch.bincount(labels, minlength=n_classes)
        n_neg, n_pos = counts[0].item(), counts[1].item()

        # print(f"  Local labels: {n_samples} total, {n_pos} positive, {n_neg} negative") # Debug print

        # Calculate weights only if both classes are present
        if n_pos > 0 and n_neg > 0:
            # Weight = Total Samples / (Num Classes * Samples in Class)
            weight_pos = n_samples / (n_classes * n_pos)
            weight_neg = n_samples / (n_classes * n_neg)
            class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float, device=device)
            weight = class_weights
            # print(f"  Using class weights: Neg={weight_neg:.2f}, Pos={weight_pos:.2f}") # Debug print
        elif n_pos == 0 and n_neg > 0:
             print("  Warning: Only negative samples in local batch. Loss might be zero.")
             # Set weight for positive to small value to avoid issues if model predicts positive
             class_weights = torch.tensor([1.0, 1e-6], dtype=torch.float, device=device)
             weight = class_weights
        elif n_neg == 0 and n_pos > 0:
             print("  Warning: Only positive samples in local batch.")
             # Set weight for negative to small value
             class_weights = torch.tensor([1e-6, 1.0], dtype=torch.float, device=device)
             weight = class_weights
        else:
             print("  Warning: No samples in local batch?")
             # Cannot compute weights if no samples

    # Define criterion inside, potentially using weights
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    # --- Local Training Loop ---
    for epoch in range(local_epochs):
        optimizer_gnn.zero_grad()
        try:
            out = local_model(data) # Forward pass

            # Ensure output and labels match shapes
            if out.shape[0] != labels.shape[0]:
                 print(f"Error: Output shape {out.shape} mismatch with labels shape {labels.shape}. Skipping backprop.")
                 continue # Skip this batch/epoch if shapes mismatch

            loss = criterion(out, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in local epoch {epoch+1}. Skipping backward pass.")
                # Potentially add more debug info here about 'out' and 'labels'
                break # Stop local training if loss becomes NaN

            loss.backward()
            optimizer_gnn.step()
            # print(f"  Local Epoch {epoch+1}/{local_epochs}, Loss: {loss.item():.4f}") # Optional print

        except Exception as e:
            print(f"Error during local training epoch {epoch+1}: {e}")
            traceback.print_exc() # Print detailed traceback
            break # Stop local training on error

    # Return the updated weights (state dictionary) on CPU
    return local_model.cpu().state_dict()

def average_weights(w_list):
    """Averages the weights (state_dicts) from a list of models."""
    if not w_list:
        return None # Return None if the list is empty

    # Check if all state_dicts have the same keys
    first_keys = w_list[0].keys()
    if not all(w.keys() == first_keys for w in w_list[1:]):
        print("Warning: Mismatched keys found in state_dicts during aggregation. Check model consistency.")
        # Fallback: return the first model's weights or handle differently
        return w_list[0]

    # Deep copy the first state_dict to store the average
    avg_w = copy.deepcopy(w_list[0])

    # Iterate through the keys (layers/parameters)
    for key in avg_w.keys():
        # Ensure the parameter is a tensor
        if torch.is_tensor(avg_w[key]):
            # Sum the tensors from all models for the current key
            for i in range(1, len(w_list)):
                if torch.is_tensor(w_list[i][key]):
                    avg_w[key] += w_list[i][key]
                else:
                    # This should ideally not happen if key check passed
                    print(f"Warning: Non-tensor found for key '{key}' in model {i} during aggregation.")
                    # Decide handling: skip key, return original, etc.
                    # For safety, maybe revert to the first model's param for this key
                    avg_w[key] = w_list[0][key]
                    break # Stop aggregating for this key

            # Divide the summed tensor by the number of models to get the average
            if torch.is_tensor(avg_w[key]):
                avg_w[key] = torch.div(avg_w[key], len(w_list))
        else:
             # If the value is not a tensor (e.g., training step counter), keep the first one
             avg_w[key] = w_list[0][key]


    return avg_w

def split_dataframe_into_clients_randomly(df, num_clients):
    """Splits a DataFrame into N chunks randomly."""
    if df.empty:
        print("Input DataFrame is empty, cannot split.")
        return []
    if num_clients <= 0:
        print("Number of clients must be positive.")
        return []

    # Shuffle the DataFrame rows
    df = df.sample(frac=1, random_state=config['seed']).reset_index(drop=True)
    # Split into N roughly equal parts
    client_dfs = np.array_split(df, num_clients)

    # Filter out any potentially empty dataframes resulting from the split
    valid_client_dfs = [cdf.copy() for cdf in client_dfs if not cdf.empty]

    print(f"Data split randomly into {len(valid_client_dfs)} non-empty clients (requested {num_clients}).")
    # Print size of each client's data
    # for i, cdf in enumerate(valid_client_dfs):
    #     print(f"  Client {i}: {len(cdf)} samples")
    return valid_client_dfs


# --- Federated Training Orchestration ---
def federated_training_ieee(global_gnn_model, dqn_agent, target_dqn, optimizer_rl, replay_buffer,
                           client_datasets, threshold_space, config,
                           start_round=0, initial_epsilon=None, initial_stopper=None):
    """Orchestrates Federated Learning with Reinforcement Learning."""
    device = config['device']
    num_actions = len(threshold_space)
    epsilon = initial_epsilon if initial_epsilon is not None else config['rl_epsilon_start']
    stopper = initial_stopper if initial_stopper is not None else EarlyStopper(patience=config['fl_early_stopping_patience'], min_delta=0.0005)

    all_round_metrics = {'f1': [], 'auc_roc': [], 'auc_pr': [], 'recall_k': [], 'loss_rl': []}
    num_features = config['num_features']

    print(f"\n--- Starting Federated Training (FL+RL) ---")
    print(f"Total Rounds: {config['fl_rounds']}, Clients/Round: {config['fl_clients_per_round']}")
    print(f"Using Feature Weighting: {config['use_feature_weighting']}")
    print(f"Using Device: {device}")
    print(f"Recall@k% Metric: Recall@{config['recall_k_percent']*100:.1f}%")

    # Move models to the designated device
    global_gnn_model.to(device)
    dqn_agent.to(device)
    target_dqn.to(device)
    target_dqn.load_state_dict(dqn_agent.state_dict()) # Initialize target DQN
    target_dqn.eval() # Target network is only used for inference

    fl_start_time = time.time()
    for t in range(start_round, config['fl_rounds']):
        round_start_time = time.time()
        global_gnn_model.train() # Set GNN to train mode for local updates
        dqn_agent.eval()         # Keep RL agent in eval mode during GNN training & initial eval

        local_weights = []
        round_metrics = {'f1': [], 'auc_roc': [], 'auc_pr': [], 'recall_k': []} # Metrics for this round

        # --- Client Selection ---
        if not client_datasets:
             print("Error: No client datasets available. Stopping training.")
             break
        num_available_clients = len(client_datasets)
        if num_available_clients == 0:
            print("Error: Client dataset list is empty. Stopping training.")
            break

        clients_this_round = min(config['fl_clients_per_round'], num_available_clients)
        if num_available_clients <= clients_this_round:
            # Select all available clients if fewer than requested/needed
            selected_indices = list(range(num_available_clients))
        else:
            # Sample clients randomly
            selected_indices = random.sample(range(num_available_clients), clients_this_round)

        print(f"\n--- FL Round {t+1}/{config['fl_rounds']} (Epsilon: {epsilon:.4f}) ---")
        print(f"Selected {len(selected_indices)} client indices: {selected_indices}")

        # --- Graph Building & State Collection for Selected Clients ---
        states_before_update = {} # Store initial state for RL
        client_graphs = {}        # Store graphs for local training & evaluation
        valid_client_indices_for_round = [] # Track clients successfully processed

        print("Building graphs and getting initial states for selected clients...")
        graph_build_total_time = 0
        for client_idx in selected_indices:
            df = client_datasets[client_idx]
            if df.empty or len(df) < 2: # Need at least 2 nodes for potential edges
                print(f"  Client {client_idx}: Skip - empty or single-node DataFrame.")
                continue
            try:
                graph_build_start = time.time()
                # Build graph (use less verbose logging within the loop)
                data = build_graph_from_cc_df(
                    df=df,
                    feature_cols=config['feature_cols'],
                    time_col=config['time_col'],
                    label_col=config['label_col'],
                    max_neighbors=config['graph_max_neighbors'],
                    time_window=config['graph_time_window'],
                    use_feature_similarity=config['graph_use_feature_similarity'],
                    similarity_threshold=config['graph_similarity_threshold'],
                    verbose=False # Less verbose inside FL loop
                )
                graph_build_time = time.time() - graph_build_start
                graph_build_total_time += graph_build_time

                # Validate graph structure
                if data.num_nodes == 0: # data.num_edges can be 0, still valid
                    print(f"  Client {client_idx}: Skip - graph built with 0 nodes.")
                    continue
                # Basic check for data consistency
                if data.x.shape[0] != data.num_nodes or data.y.shape[0] != data.num_nodes:
                     print(f"  Client {client_idx}: Skip - inconsistent node/feature/label counts after graph build.")
                     continue

                # Store graph and get initial state (embedding before local update)
                client_graphs[client_idx] = data.to(device) # Move graph to device early
                states_before_update[client_idx] = global_gnn_model.get_graph_embedding(client_graphs[client_idx])
                valid_client_indices_for_round.append(client_idx)
                # print(f"  Client {client_idx}: Graph built ({data.num_nodes} nodes, {data.num_edges} edges) in {graph_build_time:.2f}s.")

            except Exception as e:
                print(f"  Client {client_idx}: Skip - Error building graph or getting state: {e}")
                traceback.print_exc()
                continue
        print(f"Graph building for round took {graph_build_total_time:.2f}s.")

        if not valid_client_indices_for_round:
            print("Warning: No valid clients processed for this round. Skipping GNN updates and RL.")
            # Still decay epsilon? Maybe not if no experience gained.
            # epsilon = max(config['rl_epsilon_end'], epsilon * config['rl_epsilon_decay'])
            continue # Skip to next round

        # --- GNN Local Training ---
        print(f"Performing local GNN updates for {len(valid_client_indices_for_round)} clients...")
        local_update_start = time.time()
        num_successful_updates = 0
        for client_idx in valid_client_indices_for_round:
            try:
                local_w = local_update(
                    client_graphs[client_idx], # Use the already built graph
                    global_gnn_model,
                    config['fl_local_epochs'],
                    config['gnn_lr'],
                    device,
                    'y' # label attribute in PyG Data object
                )
                local_weights.append(local_w)
                num_successful_updates += 1
            except Exception as e:
                 print(f"Error during local update for client {client_idx}: {e}")
                 traceback.print_exc()
        print(f"Local updates took {time.time() - local_update_start:.2f}s for {num_successful_updates} clients.")

        # --- GNN Global Aggregation ---
        if not local_weights:
            print("Warning: No local weights collected. Skipping GNN aggregation.")
        else:
            print("Aggregating GNN weights...")
            agg_start = time.time()
            global_weights = average_weights(local_weights)
            if global_weights:
                global_gnn_model.load_state_dict(global_weights)
                print(f"Aggregation complete. Took {time.time() - agg_start:.2f}s.")
            else:
                print("Warning: Aggregation failed (average_weights returned None). Global model not updated.")

        # --- RL Experience Collection & Evaluation (using the *updated* global model) ---
        print("Evaluating updated model and collecting RL experience...")
        eval_start = time.time()
        global_gnn_model.eval() # Set GNN to eval mode for consistent evaluation
        dqn_agent.eval()         # RL agent stays in eval mode for action selection & weighting

        num_valid_clients_for_rl = 0
        temp_metrics = {'f1': [], 'auc_roc': [], 'auc_pr': [], 'recall_k': []} # Temp store per-client metrics

        for client_idx in valid_client_indices_for_round:
            # Ensure we have the graph and initial state for this client
            if client_idx not in client_graphs or client_idx not in states_before_update:
                print(f"  Client {client_idx}: Missing graph or state. Skipping RL step.")
                continue

            data = client_graphs[client_idx]
            state = states_before_update[client_idx] # State *before* local updates

            try:
                # 1. Select Action (Threshold Index)
                action_index = select_action_epsilon_greedy(state, dqn_agent, num_actions, epsilon, device)
                threshold = threshold_space[action_index]

                # 2. Get Feature Weights (if enabled)
                current_feature_weights = torch.ones(num_features, device=device) # Default to ones
                if config['use_feature_weighting']:
                    with torch.no_grad():
                         # Ensure state has batch dimension
                         state_batch = state.unsqueeze(0).to(device)
                         _, current_feature_weights = dqn_agent(state_batch)
                         current_feature_weights = current_feature_weights.squeeze(0).to(data.x.device) # Remove batch dim, move to data device

                # 3. Evaluate GNN with selected threshold and weights
                with torch.no_grad():
                    # Apply weights if enabled
                    eval_x = data.x * current_feature_weights if config['use_feature_weighting'] else data.x
                    # Create a temporary data object with potentially weighted features
                    # Ensure all necessary attributes are copied
                    temp_data = Data(x=eval_x, edge_index=data.edge_index, y=data.y,
                                     timestamp=data.timestamp, num_nodes=data.num_nodes)
                    # Already moved to device earlier: temp_data = temp_data.to(device)

                    out = global_gnn_model(temp_data) # Use updated global model
                    # Apply softmax to get probabilities for class 1 (fraud)
                    probs = torch.softmax(out, dim=1)[:, 1]
                    # Get predictions based on the selected threshold
                    pred = (probs > threshold).long()

                # 4. Calculate Metrics
                y_true_np = data.y.cpu().numpy()
                y_pred_np = pred.cpu().numpy()
                y_scores_np = probs.cpu().numpy()

                # Handle potential NaNs in scores before metric calculation
                if np.isnan(y_scores_np).any():
                     print(f"  Client {client_idx}: Warning - NaN scores detected. Replacing NaNs with 0 for metric calculation.")
                     y_scores_np = np.nan_to_num(y_scores_np, nan=0.0) # Replace NaN with 0

                f1 = 0.0; auc_roc = 0.5; auc_pr = 0.0; recall_k = 0.0 # Default values
                unique_labels = np.unique(y_true_np)

                if len(unique_labels) > 1: # Need both classes for AUC-ROC, AUC-PR
                    try: auc_roc = roc_auc_score(y_true_np, y_scores_np)
                    except ValueError: pass # Handles constant scores/labels
                    try: auc_pr = average_precision_score(y_true_np, y_scores_np)
                    except ValueError: pass
                elif np.sum(y_true_np) > 0 : # Only positive class present
                     # AUC-PR can be calculated (might be 1.0 if all scores > threshold)
                     try: auc_pr = average_precision_score(y_true_np, y_scores_np)
                     except ValueError: pass
                     auc_roc = 0.5 # Undefined, default to 0.5

                # F1 and Recall@k require positive class predictions or true positives
                # Use zero_division=0 for f1_score
                f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
                if np.sum(y_true_np) > 0: # Only calculate recall if there are true positives
                    recall_k = recall_at_k_percent(y_true_np, y_scores_np, k_percent=config['recall_k_percent'])


                temp_metrics['f1'].append(f1); temp_metrics['auc_roc'].append(auc_roc)
                temp_metrics['auc_pr'].append(auc_pr); temp_metrics['recall_k'].append(recall_k)

                # 5. Calculate Reward
                reward = calculate_reward(y_true_np, y_pred_np, y_scores_np, lambda_fpr=config['lambda_fpr'])

                # 6. Get Next State (embedding AFTER aggregation, using NON-weighted data)
                # Use the original data graph (data) to get the state reflecting the GNN update
                next_state = global_gnn_model.get_graph_embedding(data)
                done = True # Assume episode ends after each client round in this setup

                # 7. Store Experience in Replay Buffer (move states to CPU)
                replay_buffer.push(state.cpu(), action_index, reward, next_state.cpu(), done)
                num_valid_clients_for_rl += 1

                # Optional: Print per-client evaluation summary
                # print(f"  Client {client_idx}: Thr={threshold:.2f}, F1={f1:.4f}, AUC_PR={auc_pr:.4f}, Rec@{config['recall_k_percent']*100:.0f}%={recall_k:.4f}, Rew={reward:.4f}")

            except Exception as e:
                 print(f"Error during RL experience collection/evaluation for client {client_idx}: {e}")
                 traceback.print_exc()


        # Aggregate metrics for the round (handle potential NaNs/empty lists carefully)
        avg_f1 = np.nanmean([m for m in temp_metrics['f1'] if m is not None]) if temp_metrics['f1'] else 0
        avg_auc_roc = np.nanmean([m for m in temp_metrics['auc_roc'] if m is not None]) if temp_metrics['auc_roc'] else 0
        avg_auc_pr = np.nanmean([m for m in temp_metrics['auc_pr'] if m is not None]) if temp_metrics['auc_pr'] else 0
        avg_recall_k = np.nanmean([m for m in temp_metrics['recall_k'] if m is not None]) if temp_metrics['recall_k'] else 0

        all_round_metrics['f1'].append(avg_f1); all_round_metrics['auc_roc'].append(avg_auc_roc)
        all_round_metrics['auc_pr'].append(avg_auc_pr); all_round_metrics['recall_k'].append(avg_recall_k)
        print(f"Evaluation & RL data collection took {time.time() - eval_start:.2f}s.")

        # --- Decay Epsilon ---
        epsilon = max(config['rl_epsilon_end'], epsilon * config['rl_epsilon_decay'])

        # --- RL Agent Training ---
        avg_rl_loss = None
        if len(replay_buffer) >= config['rl_batch_size'] and num_valid_clients_for_rl > 0:
            print("Training RL agent...")
            rl_train_start = time.time()
            dqn_agent.train() # Set RL agent to train mode
            current_rl_loss = 0; updates_done = 0
            # Perform multiple updates per round based on collected experience
            num_rl_updates = num_valid_clients_for_rl * config['rl_updates_per_round']
            for _ in range(num_rl_updates):
                 loss_rl = train_rl_agent(dqn_agent, target_dqn, optimizer_rl, replay_buffer,
                                          config['rl_batch_size'], config['rl_gamma'], device)
                 if loss_rl is not None:
                     current_rl_loss += loss_rl
                     updates_done += 1
            if updates_done > 0:
                 avg_rl_loss = current_rl_loss / updates_done
            print(f"RL training took {time.time() - rl_train_start:.2f}s for {updates_done} updates. Avg Loss: {avg_rl_loss if avg_rl_loss is not None else 'N/A'}")
            dqn_agent.eval() # Set RL agent back to eval mode
        else:
            print("Skipping RL training (buffer too small or no valid clients).")
        all_round_metrics['loss_rl'].append(avg_rl_loss) # Append None if no training occurred

        # --- Update Target Network Periodically ---
        if (t + 1) % config['rl_target_update_freq'] == 0:
            print("Updating target DQN network...")
            target_dqn.load_state_dict(dqn_agent.state_dict())

        # --- Log Round Summary ---
        round_end_time = time.time()
        print(f"--- Round {t+1} Summary (Took {round_end_time - round_start_time:.2f}s) ---")
        print(f"Avg F1: {avg_f1:.4f} | Avg AUC-ROC: {avg_auc_roc:.4f} | Avg AUC-PR: {avg_auc_pr:.4f} | Avg Recall@{config['recall_k_percent']*100:.1f}%: {avg_recall_k:.4f}")
        print(f"Clients Processed: {len(valid_client_indices_for_round)}, GNN Updates: {num_successful_updates}, RL Exp Collected: {num_valid_clients_for_rl}")
        print(f"Replay Buffer size: {len(replay_buffer)}, Epsilon: {epsilon:.4f}")
        save_interval = 5 # Lưu mỗi 5 round
        if (t + 1) % save_interval == 0 or t == config['fl_rounds'] - 1: # Lưu ở interval hoặc round cuối
            save_checkpoint(
                gnn_model=global_gnn_model,
                rl_agent=dqn_agent,
                target_rl_agent=target_dqn,
                optimizer_rl=optimizer_rl,
                replay_buffer=replay_buffer,
                stopper=stopper,
                current_round=t, # Lưu round hiện tại đã hoàn thành
                epsilon=epsilon,
                config=config # Lưu config
            )

        # --- Check Early Stopping ---
        if stopper.step(avg_auc_pr):
            print(f"Early stopping triggered after round {t+1}.")
            # Save checkpoint cuối cùng trước khi dừng
            save_checkpoint(global_gnn_model, dqn_agent, target_dqn, optimizer_rl, replay_buffer, stopper, t, epsilon, config)
            
            # Save final models when early stopping
            print("Saving final models due to early stopping...")
            try:
                gnn_path, rl_path, metadata_path = save_final_models(
                    gnn_model=global_gnn_model,
                    rl_agent=dqn_agent,
                    config=config,
                    metrics_history=all_round_metrics
                )
                print("Final models saved due to early stopping!")
            except Exception as save_error:
                print(f"Error saving final models during early stopping: {save_error}")
            
            break
        # --- Check Early Stopping ---
        # Use a reliable metric like average AUC-PR

    fl_end_time = time.time()
    print(f"\n--- Federated Training Finished ---")
    print(f"Total FL Time: {fl_end_time - fl_start_time:.2f}s")
    
    # Save final models after training completion
    print("\n--- Saving Final Models After Training ---")
    try:
        gnn_path, rl_path, metadata_path = save_final_models(
            gnn_model=global_gnn_model,
            rl_agent=dqn_agent,
            config=config,
            metrics_history=all_round_metrics
        )
        print("Final models saved after training completion!")
    except Exception as save_error:
        print(f"Error saving final models after training: {save_error}")
    
    return all_round_metrics, global_gnn_model, dqn_agent

# --- Main Pipeline Runner ---
def run_pipeline(config):
    """Runs the complete pipeline: Data -> Oversample -> Split -> FL+RL -> Plot."""
    set_seed(config['seed'])
    device = config['device']
    print(f"Using device: {device}")
    pipeline_start_time = time.time()

    # --- 1. Load and Preprocess Data ---
    print("\n--- 1. Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(config['csv_file_path'])
        print(f"Data loaded: {df.shape}")
        # Basic check for label column
        if config['label_col'] not in df.columns:
             raise ValueError(f"Label column '{config['label_col']}' not found in CSV.")
        print("Initial class distribution:")
        print(df[config['label_col']].value_counts(normalize=True))
    except FileNotFoundError:
        print(f"Error: CSV file not found at {config['csv_file_path']}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- Scaling ---
    # Using StandardScaler, could switch to MinMaxScaler if needed
    scaler = StandardScaler()
    df_scaled = df.copy()
    try:
        # Ensure all feature columns exist before scaling
        missing_cols = [col for col in config['feature_cols'] if col not in df_scaled.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required for scaling: {missing_cols}")
        df_scaled[config['feature_cols']] = scaler.fit_transform(df_scaled[config['feature_cols']])
        print("Data scaled using StandardScaler.")
    except Exception as e:
        print(f"Error during scaling: {e}. Check config['feature_cols'].")
        exit()

    # --- Optional: Oversampling ---
    df_processed = df_scaled # Default to scaled data

    if config.get('perform_oversampling', False): # Check if flag is set and True
        print("\n--- Performing Oversampling ---")
        try:
            from imblearn.over_sampling import RandomOverSampler # Ensure import again
            X = df_scaled.drop(config['label_col'], axis=1)
            y = df_scaled[config['label_col']]
            initial_counts = Counter(y)
            print(f"Original dataset shape %s" % initial_counts)

            target_ratio = config['oversampling_target_fraud_ratio']
            n_non_fraud = initial_counts[0]
            current_fraud_count = initial_counts[1]

            # Calculate target number of fraud samples for the desired ratio
            # target_fraud / (target_fraud + n_non_fraud) = target_ratio
            # target_fraud = target_ratio * (target_fraud + n_non_fraud)
            # target_fraud * (1 - target_ratio) = target_ratio * n_non_fraud
            # target_fraud = (target_ratio / (1 - target_ratio)) * n_non_fraud
            if target_ratio < 1: # Avoid division by zero if target is 100%
                 target_fraud_count = int(np.ceil((target_ratio / (1 - target_ratio)) * n_non_fraud))
            else:
                 target_fraud_count = current_fraud_count # Should not happen for fraud usually

            print(f"Target fraud ratio: {target_ratio*100:.1f}%, Target fraud count: {target_fraud_count}")

            # Only oversample if needed and feasible
            if target_fraud_count > current_fraud_count:
                # Define sampling strategy: keep non-fraud, increase fraud
                sampling_strategy = {0: n_non_fraud, 1: target_fraud_count}
                print(f"Applying sampling strategy: {sampling_strategy}")

                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=config['seed'])
                X_res, y_res = ros.fit_resample(X, y)
                resampled_counts = Counter(y_res)
                print(f"Resampled dataset shape %s" % resampled_counts)

                # Create the resampled DataFrame
                df_resampled = pd.DataFrame(X_res, columns=X.columns)
                df_resampled[config['label_col']] = y_res

                # Shuffle the resampled data
                df_processed = df_resampled.sample(frac=1, random_state=config['seed']).reset_index(drop=True)
                print("Oversampling complete and data shuffled.")
                print("New class distribution:")
                print(df_processed[config['label_col']].value_counts(normalize=True))

            else:
                print(f"Skipping oversampling: Target count ({target_fraud_count}) not greater than current ({current_fraud_count}).")
                df_processed = df_scaled # Use original scaled data

        except ImportError:
            print("Error: `imbalanced-learn` library not found. Cannot perform oversampling.")
            print("Please install it: pip install imbalanced-learn")
            df_processed = df_scaled # Use original scaled data if import fails
        except Exception as e:
            print(f"Error during oversampling: {e}. Using original scaled data.")
            traceback.print_exc()
            df_processed = df_scaled # Use original scaled data on error
    else:
        print("\n--- Skipping Oversampling ---")


    # --- 2. Split Data into Clients ---
    print("\n--- 2. Splitting Data into Clients ---")
    # Use the potentially oversampled data (df_processed) for splitting
    clients_data_dfs = split_dataframe_into_clients_randomly(df_processed, config['num_clients'])
    if not clients_data_dfs:
        print("Error: No client data after splitting. Exiting.")
        exit()

    # --- 3. Initialize Models ---
    print("\n--- 3. Initializing Models ---")
    input_dim = config['num_features'] # Features: Time, Amount, V1-V28
    state_dim = config['rl_state_dim']  # GNN embedding size (calculated based on GCN)
    num_actions = config['threshold_num_actions'] # Discrete actions for threshold
    num_features = config['num_features'] # Needed for feature weighting head

    # Initialize GNN (using GCN)
    global_gnn_model = SimpleTSSGCNet(
        input_dim=input_dim,
        hidden_dim=config['gnn_hidden_dim'],
        num_classes=2, # Binary classification (Fraud/Non-Fraud)
        dropout_rate=config['gnn_dropout']
    ).to(device)

    # Initialize RL Agent and Target Network
    dqn_agent = DQNAgent(state_dim, num_actions, num_features).to(device)
    target_dqn = DQNAgent(state_dim, num_actions, num_features).to(device)
    target_dqn.load_state_dict(dqn_agent.state_dict()) # Copy weights
    target_dqn.eval() # Target network is for inference only

    # Initialize RL Optimizer and Replay Buffer
    optimizer_rl = optim.Adam(dqn_agent.parameters(), lr=config['rl_lr'])
    replay_buffer = ReplayBuffer(config['rl_buffer_capacity'])

    # Define the action space (threshold values)
    threshold_space = np.linspace(config['threshold_min'], config['threshold_max'], num_actions).tolist()

    print(f"Models initialized:")
    print(f"  GNN: Input Dim={input_dim}, Hidden Dim={config['gnn_hidden_dim']}")
    print(f"  RL : State Dim={state_dim}, Num Actions={num_actions}, Num Features={num_features}")
    # print(f"  GNN Model:\n{global_gnn_model}") # Optional: print model structure
    # print(f"  RL Agent Model:\n{dqn_agent}")    # Optional: print model structure
    start_round = 0
    loaded_epsilon = config['rl_epsilon_start'] # Default epsilon
    stopper = EarlyStopper(patience=config['fl_early_stopping_patience'], min_delta=0.0005) # Khởi tạo stopper mới

    checkpoint_data = load_checkpoint() # Tải checkpoint nếu có
    if checkpoint_data is not None:
        try:
            # Kiểm tra config tương thích (tuỳ chọn)
            # if checkpoint_data.get('config') != config:
            #     print("Warning: Config mismatch between checkpoint and current run. Check carefully.")

            # Load model states
            global_gnn_model.load_state_dict(checkpoint_data['gnn_model_state_dict'])
            dqn_agent.load_state_dict(checkpoint_data['rl_agent_state_dict'])
            target_dqn.load_state_dict(checkpoint_data['target_rl_agent_state_dict'])

            # Load optimizer state
            optimizer_rl.load_state_dict(checkpoint_data['optimizer_rl_state_dict'])

            # Load replay buffer
            # Cẩn thận: replay_buffer là deque đã được load trong hàm load_checkpoint
            if isinstance(checkpoint_data['replay_buffer'], deque):
                 replay_buffer.memory = checkpoint_data['replay_buffer']
                 print(f"Loaded replay buffer with {len(replay_buffer)} transitions.")
            else:
                 print("Warning: Replay buffer in checkpoint is not a deque. Starting with empty buffer.")


            # Load training state
            start_round = checkpoint_data.get('round', 0)
            loaded_epsilon = checkpoint_data.get('epsilon', config['rl_epsilon_start'])

            # Load early stopping state
            stopper.best_score = checkpoint_data.get('early_stopper_best_score', -float('inf'))
            stopper.counter = checkpoint_data.get('early_stopper_counter', 0)
            stopper.early_stop = stopper.counter >= stopper.patience # Cập nhật trạng thái stop

            print(f"Resuming training from round {start_round}")
            print(f"Loaded epsilon: {loaded_epsilon:.4f}")
            print(f"Loaded EarlyStopper state: BestScore={stopper.best_score:.4f}, Counter={stopper.counter}")

        except Exception as e:
            print(f"Error applying checkpoint data: {e}. Starting from scratch.")
            start_round = 0 # Reset về 0 nếu có lỗi khi áp dụng checkpoint
            loaded_epsilon = config['rl_epsilon_start']
            # Khởi tạo lại stopper nếu lỗi
            stopper = EarlyStopper(patience=config['fl_early_stopping_patience'], min_delta=0.0005)


    # --- 4. Run Federated Training ---
    metrics_history, final_gnn, final_rl = federated_training_ieee(
    global_gnn_model, dqn_agent, target_dqn, optimizer_rl, replay_buffer,
    clients_data_dfs, threshold_space, config,
    start_round=start_round,          # Pass the start round
    initial_epsilon=loaded_epsilon,   # Pass the loaded epsilon
    initial_stopper=stopper          # Pass the loaded stopper
)

    # --- 5. Plotting ---
    print("\n--- 5. Plotting Results ---")
    if metrics_history:
        try:
            plt.figure(figsize=(18, 10))
            plt.suptitle("FraudGNN-RL Training Metrics", fontsize=16)

            recall_k_label = f'Avg Recall@{config["recall_k_percent"]*100:.1f}%'

            # Plotting metrics that were collected
            plt.subplot(2, 3, 1)
            plt.plot(metrics_history.get('f1', []), label='Avg F1')
            plt.title('Avg F1 Score')
            plt.xlabel('Round')
            plt.ylabel('F1')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 3, 2)
            plt.plot(metrics_history.get('auc_roc', []), label='Avg AUC-ROC')
            plt.title('Avg AUC-ROC')
            plt.xlabel('Round')
            plt.ylabel('AUC-ROC')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 3, 3)
            plt.plot(metrics_history.get('auc_pr', []), label='Avg AUC-PR')
            plt.title('Avg AUC-PR')
            plt.xlabel('Round')
            plt.ylabel('AUC-PR')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 3, 4)
            plt.plot(metrics_history.get('recall_k', []), label=recall_k_label)
            plt.title(recall_k_label)
            plt.xlabel('Round')
            plt.ylabel('Recall@k')
            plt.legend()
            plt.grid(True)

            # Plot RL Loss (handle None values)
            rl_losses = [l for l in metrics_history.get('loss_rl', []) if l is not None]
            rl_rounds = [i for i, l in enumerate(metrics_history.get('loss_rl', [])) if l is not None]
            plt.subplot(2, 3, 5)
            if rl_losses:
                plt.plot(rl_rounds, rl_losses, label='RL Loss')
                plt.title('RL Training Loss')
                plt.xlabel('Round (where trained)')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            else:
                plt.title('RL Training Loss (No data)')
                plt.text(0.5, 0.5, 'Not Trained or No Data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.grid(True)

            # Placeholder for GNN Loss (if you decide to collect it)
            plt.subplot(2, 3, 6)
            plt.title('Avg GNN Loss (Not Collected)')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.text(0.5, 0.5, 'Not Collected', ha='center', va='center', transform=plt.gca().transAxes)
            plt.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            plot_filename = "fraudgnnrl_credit_card_metrics_revised.png"
            plt.savefig(plot_filename)
            print(f"Metrics plot saved to: {plot_filename}")
            # plt.show() # Uncomment to display plot directly if not in Kaggle/headless env

        except Exception as e:
            print(f"Error during plotting: {e}")
    else:
        print("No metrics history available to plot.")

    pipeline_end_time = time.time()
    print(f"\n--- Pipeline Finished ---")
    print(f"Total Pipeline Time: {pipeline_end_time - pipeline_start_time:.2f}s")
    return metrics_history, final_gnn, final_rl

try:
    # Run the entire pipeline
    metrics_history, final_gnn_model, final_rl_agent = run_pipeline(config)

    # Save final models using the dedicated function
    print("\n--- Saving Final Models ---")
    try:
        gnn_path, rl_path, metadata_path = save_final_models(
            gnn_model=final_gnn_model,
            rl_agent=final_rl_agent,
            config=config,
            metrics_history=metrics_history
        )
        print("Final models saved successfully!")
        print(f"  GNN Model: {gnn_path}")
        print(f"  RL Agent: {rl_path}")
        print(f"  Metadata: {metadata_path}")
    except Exception as save_error:
        print(f"Error saving final models: {save_error}")
        traceback.print_exc()

except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"An error occurred during pipeline execution: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    traceback.print_exc() # Print detailed error information