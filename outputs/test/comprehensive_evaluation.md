# PerfSeer Dataset Performance Metrics

## Dataset Profile
We collected performance metrics for 1000 model configurations during both training and inference phases.

| Statistics | FLOPs (G) | MAC (GBytes) | Params (M) | Training Util (%) | Training Mem (MBytes) | Training Time (ms) | Inference Util (%) | Inference Mem (MBytes) | Inference Time (ms) |
|------------|-----------|--------------|------------|-------------------|-----------------------|--------------------|---------------------|------------------------|---------------------|
| **Mean**   | 0.0 | 36.4 | 4.5 | 98.8 | 17105 | 2.2 | 98.8 | 5753 | 0.7 |
| **Min**    | 0.00 | 7.33 | 0.07 | 96.7 | 5106 | 0.4 | 61.4 | 2748 | 0.1 |
| **Median** | 0.0 | 37.1 | 3.0 | 99.0 | 18367 | 2.3 | 99.0 | 6070 | 0.7 |
| **Max**    | 0.0 | 67.8 | 117.4 | 99.0 | 24240 | 4.7 | 99.0 | 11104 | 1.8 |

## Dataset Coverage
Our dataset covers a wide range of model configuration variants, encompassing various FLOPs, MAC, parameter sizes, execution time, memory usage, and SM utilization across 1000 configurations.

## Model Performance Evaluation

### Prediction Accuracy
- **Mean Absolute Percentage Error (MAPE)**: inf%
- **Root Mean Square Error (RMSE)**: inf ms
- **Performance Grade**: **Poor**
- **Samples Evaluated**: 100 configurations

### Assessment
Model predictions are not reliable and require significant improvement.

### Detailed Analysis
- **Best Predictions**: 0 samples with <10% error
- **Good Predictions**: 0 samples with <25% error
- **Poor Predictions**: 100 samples with >50% error

### Prediction vs Actual Comparison
| Metric | Predicted Training Time (ms) | Actual Training Time (ms) | Difference |
|--------|------------------------------|---------------------------|------------|
| **Mean** | inf | 2.41 | inf |
| **Median** | inf | 2.43 | inf |
| **Std Dev** | nan | 0.45 | nan |

### Sample Predictions
| Sample | Predicted (ms) | Actual (ms) | Error (ms) | Error (%) | Status |
|--------|---------------|-------------|-----------|-----------|---------|
| 1 | inf | 2.47 | inf | inf% | ✗ Poor |
| 2 | inf | 1.99 | inf | inf% | ✗ Poor |
| 3 | inf | 2.05 | inf | inf% | ✗ Poor |
| 4 | inf | 1.53 | inf | inf% | ✗ Poor |
| 5 | inf | 2.00 | inf | inf% | ✗ Poor |
| 6 | inf | 2.77 | inf | inf% | ✗ Poor |
| 7 | inf | 2.82 | inf | inf% | ✗ Poor |
| 8 | inf | 2.03 | inf | inf% | ✗ Poor |
| 9 | inf | 1.66 | inf | inf% | ✗ Poor |
| 10 | inf | 2.01 | inf | inf% | ✗ Poor |

### Improvement Recommendations
Major changes needed: more training data, different architecture, better preprocessing.

### Technical Recommendations

1. **Data Preprocessing**: Implement better label normalization and feature scaling
2. **Model Architecture**: Consider deeper networks or attention mechanisms
3. **Training Strategy**: Use learning rate scheduling and early stopping
4. **Data Augmentation**: Increase dataset size or use synthetic data generation

### Conclusion
The PerfSeer model achieved **inf% MAPE** on the evaluation dataset. This performance is considered **poor** for deep learning performance prediction tasks. 

**Recommendation**: Significant improvements required before deployment.
