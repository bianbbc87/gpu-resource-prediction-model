# ONNX graph → op graph

import pickle
import networkx as nx

def load_compute_graph(pkl_path: str) -> nx.DiGraph:
    """
    Load computational graph from pickle file.

    The pickle file should contain a NetworkX graph
    where node["feature"] exists.
    """
    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)

    return graph
