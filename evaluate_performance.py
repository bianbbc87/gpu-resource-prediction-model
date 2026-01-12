#!/usr/bin/env python3

import os
import torch
import numpy as np
from graph.extractor import load_compute_graph, PerfGraphBuilder
from model.seernet import SeerNet

def evaluate_model_performance():
    # 설정
    config = {
        'node_dim': 30,
        'edge_dim': 5, 
        'global_dim': 18,
        'hidden_dim': 256,
        'global_node_dim': 256
    }

    # 학습된 모델 로드 (GCP 서버의 전체 학습 경로 사용)
    model = SeerNet(**config)
    model_path = "outputs/full_training/model.pt"
    if not os.path.exists(model_path):
        # 만약 전체 학습 경로가 없다면 최적화 테스트 경로 시도
        model_path = "outputs/optimized_test/model.pt"
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"✓ Loaded trained model from {model_path}")

    # 데이터 로드
    data_dir = "data/graphs"
    label_dir = "data/labels"
    files = sorted(os.listdir(data_dir))[:1000]  # 1000개 샘플로 평가

    # 메트릭 수집
    flops_list = []
    mac_list = []
    params_list = []
    train_util_list = []
    train_mem_list = []
    train_time_list = []
    infer_util_list = []
    infer_mem_list = []
    infer_time_list = []
    
    print(f"Evaluating {len(files)} samples...")
    
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f"Processing {i}/{len(files)}...")
            
        try:
            # 그래프 로드
            graph_path = os.path.join(data_dir, file)
            nx_graph = load_compute_graph(graph_path)
            
            # PerfGraph 생성 (FLOPs, MAC, Params 추출용)
            builder = PerfGraphBuilder()
            perfgraph = builder.build(nx_graph)
            
            # 실제 라벨 로드
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            with open(label_path, "r") as f:
                label_dict = eval(f.read())
                train_label = label_dict["train"]
                infer_label = label_dict["infer"]
            
            # 메트릭 파싱: time|util|mem_util|mem_usage|peak_util|peak_mem_util|peak_mem_usage
            train_metrics = train_label.split("|")
            infer_metrics = infer_label.split("|")
            
            # 그래프에서 FLOPs, MAC, Params 계산 (추정)
            total_flops = 0
            total_params = 0
            total_mac = 0
            for _, node_data in nx_graph.nodes(data=True):
                feature = node_data.get('feature', {})
                total_flops += feature.get('flops', 0)
                total_params += feature.get('memory_info', {}).get('weight_size', 0)
                total_mac += feature.get('memory_info', {}).get('bytes', 0)
            
            flops_list.append(total_flops / 1e9)  # Convert to G
            mac_list.append(total_mac / 1e9)      # Convert to GBytes  
            params_list.append(total_params / 1e6) # Convert to M
            
            train_time_list.append(float(train_metrics[0]))
            train_util_list.append(float(train_metrics[1]))
            train_mem_list.append(float(train_metrics[3]))
            
            infer_time_list.append(float(infer_metrics[0]))
            infer_util_list.append(float(infer_metrics[1]))
            infer_mem_list.append(float(infer_metrics[3]))
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # 통계 계산
    def calc_stats(data):
        arr = np.array(data)
        return {
            'mean': np.mean(arr),
            'min': np.min(arr),
            'median': np.median(arr),
            'max': np.max(arr)
        }
    
    stats = {
        'FLOPs (G)': calc_stats(flops_list),
        'MAC (GBytes)': calc_stats(mac_list), 
        'Params (M)': calc_stats(params_list),
        'Training Util (%)': calc_stats(train_util_list),
        'Training Mem (MBytes)': calc_stats(train_mem_list),
        'Training Time (ms)': calc_stats(train_time_list),
        'Inference Util (%)': calc_stats(infer_util_list),
        'Inference Mem (MBytes)': calc_stats(infer_mem_list),
        'Inference Time (ms)': calc_stats(infer_time_list)
    }
    
    # README 스타일 Markdown 테이블 생성
    markdown_content = f"""# PerfSeer Dataset Performance Metrics

## Dataset Profile
We collected performance metrics for {len(flops_list)} model configurations during both training and inference phases.

| Statistics | FLOPs (G) | MAC (GBytes) | Params (M) | Training Util (%) | Training Mem (MBytes) | Training Time (ms) | Inference Util (%) | Inference Mem (MBytes) | Inference Time (ms) |
|------------|-----------|--------------|------------|-------------------|-----------------------|--------------------|---------------------|------------------------|---------------------|
| **Mean**   | {stats['FLOPs (G)']['mean']:.1f} | {stats['MAC (GBytes)']['mean']:.1f} | {stats['Params (M)']['mean']:.1f} | {stats['Training Util (%)']['mean']:.1f} | {stats['Training Mem (MBytes)']['mean']:.0f} | {stats['Training Time (ms)']['mean']:.1f} | {stats['Inference Util (%)']['mean']:.1f} | {stats['Inference Mem (MBytes)']['mean']:.0f} | {stats['Inference Time (ms)']['mean']:.1f} |
| **Min**    | {stats['FLOPs (G)']['min']:.2f} | {stats['MAC (GBytes)']['min']:.2f} | {stats['Params (M)']['min']:.2f} | {stats['Training Util (%)']['min']:.1f} | {stats['Training Mem (MBytes)']['min']:.0f} | {stats['Training Time (ms)']['min']:.1f} | {stats['Inference Util (%)']['min']:.1f} | {stats['Inference Mem (MBytes)']['min']:.0f} | {stats['Inference Time (ms)']['min']:.1f} |
| **Median** | {stats['FLOPs (G)']['median']:.1f} | {stats['MAC (GBytes)']['median']:.1f} | {stats['Params (M)']['median']:.1f} | {stats['Training Util (%)']['median']:.1f} | {stats['Training Mem (MBytes)']['median']:.0f} | {stats['Training Time (ms)']['median']:.1f} | {stats['Inference Util (%)']['median']:.1f} | {stats['Inference Mem (MBytes)']['median']:.0f} | {stats['Inference Time (ms)']['median']:.1f} |
| **Max**    | {stats['FLOPs (G)']['max']:.1f} | {stats['MAC (GBytes)']['max']:.1f} | {stats['Params (M)']['max']:.1f} | {stats['Training Util (%)']['max']:.1f} | {stats['Training Mem (MBytes)']['max']:.0f} | {stats['Training Time (ms)']['max']:.1f} | {stats['Inference Util (%)']['max']:.1f} | {stats['Inference Mem (MBytes)']['max']:.0f} | {stats['Inference Time (ms)']['max']:.1f} |

## Dataset Coverage
Our dataset covers a wide range of model configuration variants, encompassing various FLOPs, MAC, parameter sizes, execution time, memory usage, and SM utilization across {len(flops_list)} configurations.
"""

    # 모델 성능 평가 추가
    predictions = []
    
    print(f"\nPerforming model predictions...")
    
    for i, file in enumerate(files[:100]):  # 처음 100개로 예측 테스트
        try:
            # 그래프 로드
            graph_path = os.path.join(data_dir, file)
            nx_graph = load_compute_graph(graph_path)
            
            # PerfGraph 생성
            builder = PerfGraphBuilder()
            perfgraph = builder.build(nx_graph)
            
            # 예측 (log space에서 예측하므로 exp로 변환, overflow 방지)
            with torch.no_grad():
                log_prediction = model(perfgraph).item()
                # Clamp to prevent overflow
                log_prediction = np.clip(log_prediction, -10, 10)  # e^10 ≈ 22000ms
                prediction = np.exp(log_prediction)  # Convert back from log space
            
            predictions.append(prediction)
            
        except Exception as e:
            predictions.append(0)  # 실패시 0으로 처리
            continue
    
    # 예측 성능 분석
    actual_times = train_time_list[:len(predictions)]
    predictions = np.array(predictions[:len(actual_times)])
    actual_times = np.array(actual_times)
    
    # 오차 계산
    errors = np.abs(predictions - actual_times)
    mape = np.mean(errors / (actual_times + 1e-6)) * 100
    rmse = np.sqrt(np.mean((predictions - actual_times) ** 2))
    
    # 성능 등급 판정
    if mape < 10:
        grade = "Excellent"
        assessment = "Model predictions are highly accurate and ready for production use."
        improvements = "No major improvements needed. Consider fine-tuning for edge cases."
    elif mape < 25:
        grade = "Good"
        assessment = "Model predictions are reasonably accurate for most use cases."
        improvements = "Consider increasing training data or adjusting learning rate for better accuracy."
    elif mape < 50:
        grade = "Fair"
        assessment = "Model predictions show moderate accuracy but need improvement."
        improvements = "Increase training epochs, use more data, or try different model architecture."
    else:
        grade = "Poor"
        assessment = "Model predictions are not reliable and require significant improvement."
        improvements = "Major changes needed: more training data, different architecture, better preprocessing."
    
    # 상세 분석
    prediction_analysis = f"""
## Model Performance Evaluation

### Prediction Accuracy
- **Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%
- **Root Mean Square Error (RMSE)**: {rmse:.2f} ms
- **Performance Grade**: **{grade}**
- **Samples Evaluated**: {len(predictions)} configurations

### Assessment
{assessment}

### Detailed Analysis
- **Best Predictions**: {np.sum(errors < actual_times * 0.1)} samples with <10% error
- **Good Predictions**: {np.sum(errors < actual_times * 0.25)} samples with <25% error
- **Poor Predictions**: {np.sum(errors > actual_times * 0.5)} samples with >50% error

### Prediction vs Actual Comparison
| Metric | Predicted Training Time (ms) | Actual Training Time (ms) | Difference |
|--------|------------------------------|---------------------------|------------|
| **Mean** | {np.mean(predictions):.2f} | {np.mean(actual_times):.2f} | {abs(np.mean(predictions) - np.mean(actual_times)):.2f} |
| **Median** | {np.median(predictions):.2f} | {np.median(actual_times):.2f} | {abs(np.median(predictions) - np.median(actual_times)):.2f} |
| **Std Dev** | {np.std(predictions):.2f} | {np.std(actual_times):.2f} | {abs(np.std(predictions) - np.std(actual_times)):.2f} |

### Sample Predictions
| Sample | Predicted (ms) | Actual (ms) | Error (ms) | Error (%) | Status |
|--------|---------------|-------------|-----------|-----------|---------|
"""
    
    # 샘플 예측 결과 (처음 10개)
    for i in range(min(10, len(predictions))):
        error_pct = abs(predictions[i] - actual_times[i]) / (actual_times[i] + 1e-6) * 100
        status = "✓ Good" if error_pct < 25 else "⚠ Fair" if error_pct < 50 else "✗ Poor"
        prediction_analysis += f"| {i+1} | {predictions[i]:.2f} | {actual_times[i]:.2f} | {abs(predictions[i] - actual_times[i]):.2f} | {error_pct:.1f}% | {status} |\n"
    
    prediction_analysis += f"""
### Improvement Recommendations
{improvements}

### Technical Recommendations
"""
    
    # 기술적 개선 제안
    if mape > 50:
        prediction_analysis += """
1. **Data Preprocessing**: Implement better label normalization and feature scaling
2. **Model Architecture**: Consider deeper networks or attention mechanisms
3. **Training Strategy**: Use learning rate scheduling and early stopping
4. **Data Augmentation**: Increase dataset size or use synthetic data generation
"""
    elif mape > 25:
        prediction_analysis += """
1. **Hyperparameter Tuning**: Optimize learning rate, batch size, and model dimensions
2. **Training Duration**: Increase number of epochs with proper validation
3. **Regularization**: Add dropout or weight decay to prevent overfitting
4. **Ensemble Methods**: Combine multiple models for better predictions
"""
    else:
        prediction_analysis += """
1. **Fine-tuning**: Minor adjustments to learning rate and training duration
2. **Validation**: Implement cross-validation for robust performance assessment
3. **Deployment**: Model is ready for production with monitoring systems
4. **Optimization**: Consider model compression for faster inference
"""
    
    prediction_analysis += f"""
### Conclusion
The PerfSeer model achieved **{mape:.2f}% MAPE** on the evaluation dataset. This performance is considered **{grade.lower()}** for deep learning performance prediction tasks. 

**Recommendation**: {'Deploy with confidence' if mape < 10 else 'Deploy with monitoring' if mape < 25 else 'Improve before deployment' if mape < 50 else 'Significant improvements required before deployment'}.
"""
    # 통합 리포트 생성
    full_report = markdown_content + prediction_analysis
    
    # 파일 저장 (학습 중인 경로에 맞춰 저장)
    output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comprehensive_evaluation.md")
    with open(output_path, "w") as f:
        f.write(full_report)
    
    print(f"\n✓ Comprehensive evaluation completed!")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Dataset: {len(flops_list)} configurations analyzed")
    print(f"✓ Model MAPE: {mape:.2f}% ({grade})")
    print(f"✓ Recommendation: {'Deploy ready' if mape < 25 else 'Needs improvement'}")

if __name__ == "__main__":
    evaluate_model_performance()
