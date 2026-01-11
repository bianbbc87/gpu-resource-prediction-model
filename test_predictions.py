#!/usr/bin/env python3

import os
import torch
from graph.extractor import load_compute_graph, PerfGraphBuilder
from model.seernet import SeerNet

# 설정
config = {
    'node_dim': 30,
    'edge_dim': 5, 
    'global_dim': 18,
    'hidden_dim': 256,
    'global_node_dim': 256
}

# 모델 로드 (학습된 가중치)
model = SeerNet(**config)
model.load_state_dict(torch.load("outputs/test/model.pt", map_location="cpu"))
print("✓ Loaded trained model from outputs/test/model.pt")

# 첫 번째 데이터 테스트
data_dir = "data/graphs"
label_dir = "data/labels"
files = sorted(os.listdir(data_dir))[:3]

print("=== Performance Prediction Results ===")

for i, file in enumerate(files):
    # 그래프 로드
    graph_path = os.path.join(data_dir, file)
    nx_graph = load_compute_graph(graph_path)
    
    # PerfGraph 생성
    builder = PerfGraphBuilder()
    perfgraph = builder.build(nx_graph)
    
    # 예측
    model.eval()
    with torch.no_grad():
        prediction = model(perfgraph)
    
    # 실제 라벨 로드
    label_file = os.path.splitext(file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)
    
    with open(label_path, "r") as f:
        label_dict = eval(f.read())
        train_label = label_dict["train"]
        infer_label = label_dict["infer"]
    
    # 결과 출력
    print(f"\n--- Sample {i+1}: {file[:50]}... ---")
    print(f"Predicted Performance: {prediction.item():.2f}")
    print(f"Actual Train Label: {train_label}")
    print(f"Actual Infer Label: {infer_label}")
    
    # 라벨 파싱
    train_metrics = train_label.split("|")
    if len(train_metrics) >= 1:
        actual_time = float(train_metrics[0])
        print(f"Actual Training Time: {actual_time:.2f} ms")
        print(f"Prediction Error: {abs(prediction.item() - actual_time):.2f}")
