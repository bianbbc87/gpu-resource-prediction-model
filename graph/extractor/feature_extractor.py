# node/edge/global feature 계산

from graph.perfgraph.node_features import Feature, Args, MemoryInfo
from graph.perfgraph.edge_features import extract_edge_feature
from graph.perfgraph.global_features import compute_global_feature

def extract_node_features(nx_graph):
    """
    Parse node features and collect statistics
    """
    node_features = {}
    flops_list = []
    mac_list = []
    weight_list = []

    for node in nx_graph.nodes():
        raw = nx_graph.nodes[node]["feature"]

        feat = Feature()
        feat.type = raw.get("type", "")

        # args
        args = Args()
        for k, v in raw.get("args", {}).items():
            if hasattr(args, k):
                setattr(args, k, v)
        feat.args = args

        # memory info
        mem = MemoryInfo()
        for k, v in raw.get("memory_info", {}).items():
            if hasattr(mem, k):
                setattr(mem, k, v)
        feat.memory_info = mem

        feat.flops = raw.get("flops", 0)
        feat.arith_intensity = raw.get("arith_intensity", 0)

        node_features[node] = feat

        flops_list.append(feat.flops)
        mac_list.append(mem.bytes)
        weight_list.append(mem.weight_size)

    return node_features, flops_list, mac_list, weight_list

def extract_edge_features(nx_graph):
    """
    Extract edge features and topology
    """
    edge_index = []
    edge_features = []
    edge_tensor_sizes = []

    for src, dst in nx_graph.edges():
        edge_index.append([src, dst])
        e = extract_edge_feature(nx_graph, src, dst)
        edge_features.append(e)
        edge_tensor_sizes.append(e[0])  # tensor size

    return edge_index, edge_features, edge_tensor_sizes

def extract_global_features(
    flops_list,
    mac_list,
    weight_list,
    edge_tensor_sizes,
    num_nodes,
    num_edges,
    batch_size,
):
    """
    Compute global feature vector u
    """
    return compute_global_feature(
        flops_list=flops_list,
        mac_list=mac_list,
        weight_list=weight_list,
        edge_tensor_sizes=edge_tensor_sizes,
        num_nodes=num_nodes,
        num_edges=num_edges,
        batch_size=batch_size,
    )

