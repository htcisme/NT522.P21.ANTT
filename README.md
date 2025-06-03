# NT522.P21.ANTT - Machine Learning in Information Security

**FraudGNN-RL: Federated Credit Card Fraud Detection System**

A comprehensive project implementing a federated learning-based credit card fraud detection system using Graph Neural Networks (GNNs) and Reinforcement Learning (RL) for enhanced privacy-preserving fraud detection.

## 👥 Team Members

1. **Nguyen Dinh Khang**
2. **Hoang Bao Phuoc**
3. **Do Quang Trung**

## 📋 Project Overview

This project implements an advanced fraud detection system that combines:
- **Federated Learning (FL)**: Enables collaborative learning across multiple financial institutions without sharing raw customer data
- **Graph Neural Networks (GNNs)**: Captures complex spatial and temporal relationships between transactions
- **Reinforcement Learning (RL)**: Adaptively optimizes detection thresholds and feature weights
- **Privacy-Preserving Design**: Maintains data confidentiality while achieving high detection performance

## 🎯 Objectives

- **Privacy Protection**: Use Federated Learning to avoid sharing raw data between banks
- **Complex Pattern Detection**: Apply GNNs to capture temporal and spatial transaction relationships
- **Adaptive Optimization**: Use RL for automatic threshold and feature weight adjustment
- **Imbalanced Data Handling**: Address the low fraud rate challenge in real-world scenarios

## 🏗️ System Architecture

### 1. **Federated Learning Framework**
```
Central Server ← → Multiple Clients (Banks)
├── Model Aggregation (FedAvg)
├── Global Model Distribution
└── Privacy-Preserving Training
```

### 2. **Graph Neural Network Architecture**
```
SimpleTSSGCNet:
├── Spatial Modeling (GCN)
│   ├── GCN Layer 1 + BatchNorm + ReLU
│   └── GCN Layer 2 + BatchNorm + ReLU
├── Temporal Modeling (GRU)
│   ├── Time-aware Attention
│   └── GRU Processing
└── Fusion & Classification
    ├── Concatenate [Spatial + Temporal]
    ├── Dropout
    └── Linear Output Layer
```

### 3. **Reinforcement Learning Agent**
```
DQN Agent:
├── Shared Network (128-dim hidden)
├── Q-Value Head (Threshold Selection)
└── Feature Weight Head (Feature Importance)
```

## 📊 Dataset and Preprocessing

### Dataset Characteristics
- **Source**: Credit Card Fraud Detection Dataset
- **Features**: Time, Amount, V1-V28 (PCA components), Class
- **Challenge**: Highly imbalanced (~0.17% fraud transactions)
- **Size**: 284,807 transactions

### Graph Construction Strategy
- **Nodes**: Individual transactions
- **Edge Criteria**:
  - **Temporal proximity**: Within 1-hour time window
  - **Feature similarity**: Cosine similarity > 0.9 threshold
  - **Max neighbors**: 30 transactions per node

### Data Preprocessing Pipeline
1. **Feature Standardization**: StandardScaler for all numerical features
2. **Optional Oversampling**: RandomOverSampler to achieve 2% fraud ratio
3. **Client Distribution**: Random split across federated clients
4. **Graph Construction**: Build transaction relationship graphs

## 🔧 Key Components

### 1. **Graph Builder** (`build_graph_from_cc_df`)
- Constructs transaction graphs from raw data
- Connects nodes based on temporal and feature similarity
- Optimized batch processing for efficiency

### 2. **GNN Model** (`SimpleTSSGCNet`)
- **Spatial Component**: GCN layers for spatial relationship learning
- **Temporal Component**: GRU with time-aware attention mechanism
- **Output**: Node embeddings for classification

### 3. **RL Agent** (`DQNAgent`)
- **State Space**: Graph embeddings from GNN
- **Action Space**: Threshold selection + Feature weighting
- **Reward Function**: F1 score - λ × False Positive Rate

### 4. **Federated Training** (`federated_training_ieee`)
- Orchestrates FL training rounds
- Manages local training on each client
- Performs global model aggregation
- Collects RL experience and updates agent

## 📈 Evaluation Metrics

### Primary Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-PR**: Area Under Precision-Recall Curve
- **AUC-ROC**: Area Under ROC Curve
- **Recall@k%**: Recall within top k% predictions

### Reward Function Design
```python
reward = f1_score - λ × false_positive_rate
```
Where λ = 0.3 balances fraud detection vs. false alarm reduction.

## ⚙️ Configuration Parameters

### Model Hyperparameters
```python
config = {
    # GNN Settings
    'gnn_hidden_dim': 64,
    'gnn_dropout': 0.5,
    'gnn_lr': 1e-3,
    
    # RL Settings
    'rl_lr': 1e-4,
    'rl_epsilon_start': 1.0,
    'rl_epsilon_decay': 0.998,
    'rl_gamma': 0.99,
    
    # Federated Learning
    'num_clients': 10,
    'fl_rounds': 100,
    'fl_clients_per_round': 3,
    'fl_local_epochs': 1,
    
    # Graph Construction
    'graph_max_neighbors': 30,
    'graph_time_window': 3600,  # 1 hour
    'graph_similarity_threshold': 0.9,
}
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

### Data Setup
1. Download Credit Card Fraud Detection Dataset
2. Update path in `config['csv_file_path']`

### Running the System
```python
python CreditCardFraud.py
```

### Resume Training
- Checkpoints automatically saved every 5 rounds
- System auto-loads latest checkpoint on restart

## 📁 Project Structure

```
NT522.P21.ANTT/
├── LICENSE                    # Apache 2.0 License
├── README.md                 # This documentation
├── CreditCardFraud.py        # Main implementation
├── models/                   # Saved models directory
│   ├── final_gnn_model.pth
│   ├── final_rl_agent.pth
│   └── model_metadata.json
├── checkpoints/              # Training checkpoints
│   └── checkpoint.pth
├── fraudgnnrl_metrics.png    # Training visualization
├── report/                   # Project documentation
│   ├── poster.pdf           # Academic poster
│   ├── report.pdf           # Comprehensive report
│   └── slides.pdf           # Presentation slides
└── evaluation/              # Individual evaluations
    ├── dinhkhang.pdf
    ├── hoangphuc.pdf
    └── quangtrung.pdf
```

## 🔄 Training Pipeline

1. **Data Loading & Preprocessing**
   - Load and preprocess transaction data
   - Apply feature scaling and optional oversampling

2. **Federated Environment Setup**
   - Distribute data across simulated clients
   - Initialize GNN and RL models

3. **Training Loop**
   ```
   For each FL round:
   ├── Select random subset of clients
   ├── Build transaction graphs for clients
   ├── Perform local GNN training
   ├── Aggregate global model weights (FedAvg)
   ├── Collect RL experience from predictions
   ├── Train RL agent with collected experience
   └── Update target networks periodically
   ```

4. **Model Evaluation & Saving**
   - Generate performance metrics
   - Save trained models and metadata
   - Create training visualization plots

## 🎯 Expected Results

- **High F1 Score**: Balanced precision-recall performance
- **Low False Positive Rate**: Minimized false alarms
- **Good Recall@1%**: Detect majority of fraud in top 1% suspicious transactions
- **Privacy Preservation**: No raw data sharing between clients
- **Adaptive Performance**: RL-driven threshold optimization

## 🔧 Special Features

### 1. **Robust Checkpointing System**
- Automatic checkpoint saving every 5 rounds
- Complete training state preservation
- Seamless training resumption

### 2. **Early Stopping Mechanism**
- Monitors AUC-PR with patience=15 rounds
- Prevents overfitting and saves computational resources

### 3. **Adaptive Learning**
- RL agent automatically adjusts detection thresholds
- Dynamic feature weighting for optimal performance

### 4. **Error Handling**
- Graceful handling of edge cases (empty graphs, NaN values)
- Robust system degradation under adverse conditions

## 📚 Technical Innovations

### Graph-based Transaction Modeling
- Novel approach to represent transactions as graph nodes
- Temporal and similarity-based edge construction
- Captures complex fraud patterns through graph structure

### Federated Privacy-Preserving Learning
- Enables collaborative fraud detection across institutions
- Maintains strict data privacy requirements
- Aggregates knowledge without exposing sensitive information

### Reinforcement Learning Optimization
- Adaptive threshold selection for dynamic fraud patterns
- Feature importance weighting for improved detection
- Continuous learning from detection outcomes

## 📖 Documentation

### Academic Reports
- **[report.pdf](report/report.pdf)**: Comprehensive technical documentation
- **[slides.pdf](report/slides.pdf)**: Project presentation materials
- **[poster.pdf](report/poster.pdf)**: Academic poster summary

### Individual Contributions
- **[dinhkhang.pdf](evaluation/dinhkhang.pdf)**: Dinh Khang's evaluation
- **[hoangphuc.pdf](evaluation/hoangphuc.pdf)**: Hoang Phuc's evaluation  
- **[quangtrung.pdf](evaluation/quangtrung.pdf)**: Quang Trung's evaluation

## 🏫 Course Information

- **Course Code**: NT522.P21.ANTT
- **Course Title**: Machine Learning in Information Security
- **Academic Year**: 2024-2025
- **Institution**: [University Name]

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- Graph Neural Networks for Fraud Detection
- Federated Learning Frameworks and Applications
- Deep Reinforcement Learning for Adaptive Systems
- Credit Card Fraud Detection Benchmarks and Datasets

---

**Disclaimer**: This is an academic research project for educational purposes. Production deployment would require additional validation, security measures, and regulatory compliance considerations.

**Contact**: For questions about this project, please refer to the individual evaluation documents for team member contact information.