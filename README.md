# FraudGNN-RL: A Graph Neural Network with Reinforcement Learning for Adaptive Financial Fraud Detection

FraudGNN-RL is a cutting-edge financial fraud detection system that integrates **Graph Neural Networks (GNN)**, **Reinforcement Learning (RL)**, and **Federated Learning (FL)** to detect fraudulent transactions in credit card datasets. By modeling transactions as graphs, optimizing detection thresholds with RL, and preserving privacy via FL, the system achieves high recall and robust performance on imbalanced datasets.

---

## 👥 Team Members

| Name              | Student ID | Role   |
|-------------------|------------|--------|
| **Do Quang Trung**    | 23521673   | Leader |
| **Nguyen Dinh Khang** | 23520694   | Member |
| **Hoang Bao Phuoc**   | 23521231   | Member |

  
---

## 🏫 Course Information
- **Course Code**: NT522.P21.ANTT
- **Course Title**: Machine Learning in Information Security
- **Academic Year**: 2024-2025
- **Institution**: University of Information Technology, Vietnam National University Ho Chi Minh City (UIT-VNUHCM)

---

## 📋 Project Overview
FraudGNN-RL addresses the challenge of detecting financial fraud in highly imbalanced datasets. Key features include:
1. **Graph-Based Modeling**: Transactions are represented as graphs with PCA-transformed features (V1-V28) as node attributes.
2. **GNN for Feature Learning**: A Temporal Self-Supervised Graph Convolutional Network (TSSGC) captures transaction patterns.
3. **RL for Adaptive Thresholding**: A Deep Q-Network with Normalized Advantage Function (DQN-NAF) optimizes detection thresholds.
4. **Federated Learning**: FedAvg ensures privacy-preserving model training across multiple clients.
5. **Error Handling**: Robust handling of edge cases like empty graphs and NaN values.

The system outperforms traditional methods (e.g., XGBoost, Isolation Forest) in metrics like **Recall@5%**, **F1 Score**, **AUC-ROC**, and **AUC-PR**.

---

## 🗃️ Dataset Characteristics
- **Source**: Credit Card Fraud Detection Dataset (2023)
- **Features**: Time, Amount, V1-V28 (anonymized PCA components), Class
- **Size**: 550,000 transactions
- **Challenge**: Highly imbalanced (~0.172% fraud transactions, 946 fraudulent transactions)
- **Access**: Available on Kaggle ([link](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023))

---

## 🏗️ System Architecture
The FraudGNN-RL pipeline includes:
1. **Data Preprocessing**: Transaction data is converted into graphs with nodes (transactions) and edges (similarity-based connections).
2. **GNN Module**: TSSGC learns node embeddings for fraud classification.
3. **RL Module**: DQN-NAF dynamically adjusts detection thresholds to maximize **F1 - 0.3 × FPR**.
4. **FL Module**: FedAvg aggregates local models from multiple clients without sharing raw data.
5. **Evaluation**: Metrics include F1, AUC-ROC, AUC-PR, and Recall@5%.

![Architecture Diagram](img/Federated%20GNN%20Based%20Fraud%20Detection%20Pipeline%20with%20RL%20Driven%20Adaptation.png)

---

## 📁 Project Structure
```
NT522.../
├── img/                      # Image files (if any)
├── report/                   # Report and evaluation files
│   ├── evaluation/           # Evaluation documents
│   │   ├── dinhkhang.pdf
│   │   ├── hoangphuoc.pdf
│   │   └── quangtrung.pdf
│   ├── poster.pdf            # Project poster
│   ├── report.pdf            # Detailed project report
│   └── slides.pdf            # Presentation slides
├── CreditCardFraud.py        # Main implementation for Kaggle
└── LICENSE                   # Project license
```

**Note**: On Kaggle, the dataset (`creditcard.csv`) is accessed directly from the input directory (e.g., `/kaggle/input/credit-card-fraud-detection-dataset-2023/creditcard.csv`).

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.8+ (pre-installed on Kaggle)
- **Hardware**: GPU with CUDA support available on Kaggle
- **Dependencies** (pre-installed or installable on Kaggle):
  ```bash
  pip install torch==1.8.0 torch-geometric==2.0.1 pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
  ```

### Installation
1. Upload the project files to a Kaggle notebook:
   - Create a new Kaggle notebook.
   - Upload `CreditCardFraud.py`.
2. Add the dataset:
   - Import the "Credit Card Fraud Detection Dataset 2023" from Kaggle ([link](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)) to your notebook.

### Data Setup
- The dataset is automatically available in the Kaggle input directory.
- Update the dataset path in `CreditCardFraud.py` if needed:
  ```python
  config['csv_file_path'] = '/kaggle/input/credit-card-fraud-detection-dataset-2023/creditcard.csv'
  ```

### Running the System
1. Open the Kaggle notebook and add `CreditCardFraud.py` as the main script.
2. Run the notebook:
   - Execute all cells to train and evaluate the model.
   - Checkpoints and models are saved in the Kaggle output directory (e.g., `/kaggle/working/`).
3. Results (metrics and visualizations) are logged to the notebook output and saved in the working directory.

---

## 🔧 Configuration Parameters
Key configurations in `CreditCardFraud.py`:
```python
config = {
    # GNN Settings
    'gnn_hidden_dim': 64,
    'gnn_dropout': 0.3,
    'gnn_lr': 1e-3,
    'graph_max_neighbors': 10,
    
    # RL Settings
    'threshold_num_actions': 100,
    'rl_lr': 1e-4,
    'rl_batch_size': 32,
    
    # FL Settings
    'num_clients': 5,
    'fl_rounds': 10,
    
    # Training Settings
    'num_epochs': 50,
    'batch_size': 128,
    'checkpoint_save_path': '/kaggle/working/checkpoint.pth'
}
```

---

## 📊 Primary Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-PR**: Area Under Precision-Recall Curve
- **AUC-ROC**: Area Under ROC Curve
- **Recall@5%**: Recall within top 5% predictions

---

## 🎯 Expected Results
FraudGNN-RL achieves state-of-the-art performance:
- **F1**: 0.8957
- **AUC-ROC**: 0.9771
- **AUC-PR**: 0.8752
- **Recall@5%**: 91.51 

### Detailed Results
| Method          | AUC-ROC | AUC-PR | F1 Score | Recall@1% |
|-----------------|---------|--------|----------|-----------|
| XGBoost         | 0.9570  | 0.4380 | 0.7830   | 72.10    |
| Isolation Forest| 0.9350  | 0.3920 | 0.7450   | 67.20    |
| LOF             | 0.9220  | 0.3750 | 0.7120   | 64.80    |
| DeepAE          | 0.9720  | 0.5380 | 0.8520   | 81.50    |
| **Our Methods** | **0.9774** | **0.8752** | **0.8957** | **91.51** |

---

## 🛠️ Troubleshooting
- **File Not Found Error**: Ensure the dataset is imported and `config['csv_file_path']` points to `/kaggle/input/credit-card-fraud-detection-dataset-2023/creditcard.csv`.
- **Checkpoint Loading Failure**: Verify `config['checkpoint_save_path']` matches the Kaggle working directory (e.g., `/kaggle/working/checkpoint.pth`).
- **NaN Metrics**: Check for imbalanced client data; enable oversampling via `imbalanced-learn`.
- **Memory Issues**: Reduce `graph_max_neighbors` or `num_clients` for resource-constrained Kaggle sessions.

---

## ⚠️ Limitations and Future Work
### Limitations
- Lower AUC-PR compared to GCN due to RL prioritization of Recall@5%.
- High computational cost in federated settings.
- Limited interpretability due to PCA-transformed features.

### Future Work
- Integrate Explainable AI (e.g., SHAP, GNNExplainer) for better interpretability.
- Optimize computational efficiency via model compression.
- Extend to online learning for real-time fraud detection.

---

## 🔗 References
- CUI, Yiwen, et al. FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection. IEEE Open Journal of the Computer Society, 2025.

---

## 📬 Contact
For questions, contact the team via [team.email@example.com] or refer to [report.pdf](report/report.pdf) and [poster.pdf](report/poster.pdf) for individual evaluation details.
