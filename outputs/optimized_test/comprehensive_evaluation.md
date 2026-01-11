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
- **Mean Absolute Percentage Error (MAPE)**: 38.26%
- **Root Mean Square Error (RMSE)**: 1.07 ms
- **Performance Grade**: **Fair**
- **Samples Evaluated**: 100 configurations

### Assessment
Model predictions show moderate accuracy but need improvement.

### Detailed Analysis
- **Best Predictions**: 6 samples with <10% error
- **Good Predictions**: 16 samples with <25% error
- **Poor Predictions**: 18 samples with >50% error

### Prediction vs Actual Comparison
| Metric | Predicted Training Time (ms) | Actual Training Time (ms) | Difference |
|--------|------------------------------|---------------------------|------------|
| **Mean** | 1.44 | 2.41 | 0.97 |
| **Median** | 1.38 | 2.43 | 1.05 |
| **Std Dev** | 0.19 | 0.45 | 0.26 |

### Sample Predictions
| Sample | Predicted (ms) | Actual (ms) | Error (ms) | Error (%) | Status |
|--------|---------------|-------------|-----------|-----------|---------|
| 1 | 1.28 | 2.47 | 1.20 | 48.4% | ⚠ Fair |
| 2 | 1.25 | 1.99 | 0.75 | 37.5% | ⚠ Fair |
| 3 | 1.25 | 2.05 | 0.81 | 39.3% | ⚠ Fair |
| 4 | 1.23 | 1.53 | 0.31 | 20.1% | ✓ Good |
| 5 | 1.24 | 2.00 | 0.76 | 37.9% | ⚠ Fair |
| 6 | 1.28 | 2.77 | 1.49 | 53.8% | ✗ Poor |
| 7 | 1.28 | 2.82 | 1.54 | 54.7% | ✗ Poor |
| 8 | 1.24 | 2.03 | 0.78 | 38.6% | ⚠ Fair |
| 9 | 1.23 | 1.66 | 0.43 | 25.9% | ⚠ Fair |
| 10 | 1.24 | 2.01 | 0.76 | 38.1% | ⚠ Fair |

### Improvement Recommendations
Increase training epochs, use more data, or try different model architecture.

### Technical Recommendations

1. **Hyperparameter Tuning**: Optimize learning rate, batch size, and model dimensions
2. **Training Duration**: Increase number of epochs with proper validation
3. **Regularization**: Add dropout or weight decay to prevent overfitting
4. **Ensemble Methods**: Combine multiple models for better predictions

### Conclusion
The PerfSeer model achieved **38.26% MAPE** on the evaluation dataset. This performance is considered **fair** for deep learning performance prediction tasks. 

**Recommendation**: Improve before deployment.
