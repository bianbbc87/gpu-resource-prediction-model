def extract_edge_feature(graph, src, dst):
    """
    Edge feature e_j
    - tensor size
    - tensor shape (B, C, H, W)
    """

    src_mem = graph.nodes[src]["feature"]["memory_info"]
    dst_mem = graph.nodes[dst]["feature"]["memory_info"]

    # tensor size (bytes)
    tensor_size = src_mem.get("output_size", 0)

    # tensor shape
    batch = src_mem.get("batch_size", 0)
    channel = src_mem.get("output_channels", 0)
    h = src_mem.get("output_h", 0)
    w = src_mem.get("output_w", 0)

    return [
        tensor_size,
        batch,
        channel,
        h,
        w,
    ]
