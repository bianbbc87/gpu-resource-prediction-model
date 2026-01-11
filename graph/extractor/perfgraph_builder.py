# PerfGraph(u, V, E) 생성

import torch

from graph.perfgraph.perfgraph import PerfGraph
from .feature_extractor import (
    extract_node_features,
    extract_edge_features,
    extract_global_features,
)

class PerfGraphBuilder:
    """
    Build PerfGraph G = (u, V, E)
    """

    def build(self, nx_graph):
        # 1. node features
        (
            node_feature_map,
            flops_list,
            mac_list,
            weight_list,
        ) = extract_node_features(nx_graph)

        num_nodes = nx_graph.number_of_nodes()
        num_edges = nx_graph.number_of_edges()

        # batch size (assume uniform)
        batch_size = (
            next(iter(node_feature_map.values()))
            .memory_info.batch_size
            if num_nodes > 0
            else 0
        )

        # 2. edge features
        edge_index, edge_features, edge_tensor_sizes = extract_edge_features(
            nx_graph
        )

        # 3. global feature
        u = extract_global_features(
            flops_list=flops_list,
            mac_list=mac_list,
            weight_list=weight_list,
            edge_tensor_sizes=edge_tensor_sizes,
            num_nodes=num_nodes,
            num_edges=num_edges,
            batch_size=batch_size,
        )

        # 4. node proportions (v_p)
        flops_total = sum(flops_list) or 1
        mac_total = sum(mac_list) or 1
        weight_total = sum(weight_list) or 1

        V = []
        for node in nx_graph.nodes():
            f = node_feature_map[node]
            f.flops_ratio = f.flops / flops_total
            f.mac_ratio = f.memory_info.bytes / mac_total
            f.weight_ratio = f.memory_info.weight_size / weight_total
            V.append(f.to_vector())

        # 5. tensorize
        V = torch.tensor(V, dtype=torch.float)
        E = torch.tensor(edge_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        u = torch.tensor(u, dtype=torch.float)

        # 6. 수치 안정성을 위한 로그 정규화 (log1p)
        # - 거대 피처(조 단위 FLOPs, 메모리 바이트)를 십 단위로 압축하여 모델 폭주를 방지합니다.
        # - 0인 데이터도 log(1+0)=0으로 안전하게 처리하며 수치적 안정성을 제공합니다.
        V[:, 23:40] = torch.log1p(V[:, 23:40]) 
        
        # E: 텐서 크기(0번 인덱스) 정규화
        E[:, 0] = torch.log1p(E[:, 0])
        
        # u: FLOPs, 메모리 통계 등 주요 지표 정규화
        u[3:16] = torch.log1p(u[3:16])

        return PerfGraph(
            u=u,
            V=V,
            E=E,
            edge_index=edge_index,
        )