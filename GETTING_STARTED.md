# PerfSeer Getting Started Guide

This guide will walk you through setting up and running PerfSeer from scratch to reproduce the paper results.

## Prerequisites

- Python 3.8+ 
- macOS/Linux system
- At least 8GB RAM
- 10GB free disk space

## Step 1: Environment Setup

### 1.1 Clone and Navigate to Project
```bash
cd /path/to/your/project/PerfSeer
```

### 1.2 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Upgrade pip
```bash
python -m pip install --upgrade pip
```

## Step 2: Install Dependencies

### 2.1 Install Base Requirements
```bash
pip install -r requirements.txt
```

### 2.2 Install PyTorch
```bash
pip install torch torchvision
```

### 2.3 Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import networkx; print('NetworkX installed successfully')"
```

## Step 3: Data Preparation

### 3.1 Verify Data Structure
Ensure your data directory structure looks like this:
```
PerfSeer/
├── data/
│   ├── graphs/          # 53k+ .pkl files
│   └── labels/          # 53k+ .txt files
├── configs/
├── model/
└── train/
```

### 3.2 Check Data Availability
```bash
ls data/graphs | head -5    # Should show .pkl files
ls data/labels | head -5    # Should show .txt files
echo "Graph files: $(ls data/graphs | wc -l)"
echo "Label files: $(ls data/labels | wc -l)"
```

## Step 4: Configuration Setup

### 4.1 Create Test Configuration
Create `configs/quick_test.yaml`:
```yaml
model:
  node_dim: 30
  edge_dim: 5
  global_dim: 18
  hidden_dim: 256
  global_node_dim: 256

train:
  epochs: 5
  lr: 0.00001
  loss: mse
  batch_size: 16

dataset:
  graph_dir: data/graphs
  label_dir: data/labels
  max_samples: 1000  # Start small for testing

device:
  device: cpu  # Change to 'cuda' if GPU available
```

### 4.2 Create Full Training Configuration
Create `configs/full_training.yaml`:
```yaml
model:
  node_dim: 30
  edge_dim: 5
  global_dim: 18
  hidden_dim: 256
  global_node_dim: 256

train:
  epochs: 50
  lr: 0.00001
  loss: mse
  batch_size: 32

dataset:
  graph_dir: data/graphs
  label_dir: data/labels
  # max_samples: null  # Use all data

device:
  device: cuda  # Use GPU for full training
```

## Step 5: Quick Test Run

### 5.1 Test with Small Dataset
```bash
source venv/bin/activate
CONFIG_PATH=configs/quick_test.yaml OUTPUT_DIR=outputs/quick_test bash scripts/train.sh
```

**Expected Output:**
```
====================================
Starting training
====================================
INFO:PerfSeer:Auto-detected dimensions: node_dim=30, edge_dim=5, global_dim=18
INFO:PerfSeer:--- Epoch 001 ---
INFO:PerfSeer:Batch 100/1000, Loss: [decreasing values]
INFO:PerfSeer:[Epoch 001] Loss=X.XXXXXX | MAPE=XX.XXXX | RMSPE=XX.XXXX
...
INFO:PerfSeer:===== Training Finished =====
```

### 5.2 Verify Test Results
```bash
ls outputs/quick_test/
# Should show: model.pt, metrics.json
```

## Step 6: Performance Evaluation

### 6.1 Run Comprehensive Evaluation
```bash
source venv/bin/activate
python evaluate_performance.py
```

### 6.2 Check Results
```bash
cat outputs/test/comprehensive_evaluation.md
```

**Success Indicators:**
- MAPE < 50% (Fair performance)
- MAPE < 25% (Good performance) 
- MAPE < 10% (Excellent performance)
- No `inf` values in predictions

## Step 7: Full Training (Optional)

### 7.1 Run Full Dataset Training
```bash
source venv/bin/activate
CONFIG_PATH=configs/full_training.yaml OUTPUT_DIR=outputs/full_training bash scripts/train.sh
```

**Note:** This may take several hours depending on your hardware.

### 7.2 Monitor Training Progress
```bash
tail -f outputs/full_training/training.log  # If logging is enabled
```

## Step 8: Results Analysis

### 8.1 Generate Performance Report
```bash
# Update evaluate_performance.py to use full training model
sed -i 's/outputs\/test\/model.pt/outputs\/full_training\/model.pt/g' evaluate_performance.py
python evaluate_performance.py
```

### 8.2 Compare with Paper Results
The generated report in `outputs/full_training/comprehensive_evaluation.md` should show:
- Dataset statistics matching README table
- Model performance metrics
- Improvement recommendations

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you see "ModuleNotFoundError"
pip install -r requirements.txt
pip install torch torchvision
```

#### 2. CUDA Issues
```bash
# If CUDA not available, use CPU
# Change device: cuda to device: cpu in config files
```

#### 3. Memory Issues
```bash
# Reduce batch size or max_samples in config
# Example: batch_size: 8, max_samples: 500
```

#### 4. Infinite Loss Values
```bash
# Reduce learning rate in config
# Example: lr: 0.000001
```

#### 5. Data Loading Errors
```bash
# Verify data paths
ls data/graphs | head -5
ls data/labels | head -5
```

### Performance Expectations

| Dataset Size | Expected MAPE | Training Time | Hardware |
|--------------|---------------|---------------|----------|
| 1,000 samples | 30-50% | 5-10 minutes | CPU |
| 10,000 samples | 20-35% | 30-60 minutes | CPU |
| 53,000+ samples | 10-25% | 2-6 hours | GPU |

## Success Criteria

✅ **Setup Complete** when:
- Virtual environment activated
- All dependencies installed
- Test training completes without errors
- Performance evaluation generates report

✅ **Model Working** when:
- MAPE < 50% on test dataset
- No `inf` or `nan` values in predictions
- Loss decreases across epochs

✅ **Paper Reproduction** when:
- MAPE < 25% on full dataset
- Performance metrics match paper ranges
- Model generalizes to unseen data

## Next Steps

1. **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs
2. **Architecture Experiments**: Modify model dimensions
3. **Data Analysis**: Explore dataset characteristics
4. **Performance Optimization**: Use GPU, larger batch sizes
5. **Production Deployment**: Export model for inference

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your environment matches prerequisites
3. Start with small dataset (1000 samples) before full training
4. Monitor loss values - they should decrease over epochs

## File Structure After Setup

```
PerfSeer/
├── venv/                    # Virtual environment
├── data/
│   ├── graphs/             # Input graph data
│   └── labels/             # Performance labels
├── configs/
│   ├── quick_test.yaml     # Test configuration
│   └── full_training.yaml  # Production configuration
├── outputs/
│   ├── quick_test/         # Test results
│   └── full_training/      # Full training results
├── model/                  # Model architecture
├── train/                  # Training utilities
├── evaluate_performance.py # Evaluation script
└── requirements.txt        # Dependencies
```

This completes the setup process. You should now be able to reproduce the PerfSeer paper results!
