import numpy as np


def compute_global_feature(
    flops_list,
    mac_list,
    weight_list,
    edge_tensor_sizes,
    num_nodes,
    num_edges,
    batch_size,
):
    """
    Global feature u
    """

    # u_gp
    density = (
        num_edges / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0.0
    )

    # u_c (FLOPs statistics)
    flops_total = sum(flops_list)
    flops_mean = np.mean(flops_list)
    flops_median = np.median(flops_list)
    flops_max = max(flops_list)

    # u_m (memory statistics)
    mac_total = sum(mac_list)
    mac_mean = np.mean(mac_list)
    mac_median = np.median(mac_list)
    mac_max = max(mac_list)

    weight_total = sum(weight_list)
    weight_mean = np.mean(weight_list)
    weight_median = np.median(weight_list)
    weight_max = max(weight_list)

    edge_tensor_size_mean = (
        np.mean(edge_tensor_sizes) if edge_tensor_sizes else 0.0
    )

    # u_a
    global_arith_intensity = (
        flops_total / mac_total if mac_total > 0 else 0.0
    )

    return [
        # u_gp
        num_nodes,
        num_edges,
        density,

        # u_c
        flops_total,
        flops_mean,
        flops_median,
        flops_max,

        # u_m
        mac_total,
        mac_mean,
        mac_median,
        mac_max,
        weight_total,
        weight_mean,
        weight_median,
        weight_max,
        edge_tensor_size_mean,

        # u_a
        global_arith_intensity,

        # u_b
        batch_size,
    ]
