# PerfGraph → PyTorch dataset

import os
from torch.utils.data import Dataset
from graph.extractor import load_compute_graph, PerfGraphBuilder


class PerfGraphDataset(Dataset):
    """
    Dataset of (PerfGraph, label)
    """

    def __init__(self, graph_dir, label_dir):
        self.graph_dir = graph_dir
        self.label_dir = label_dir
        self.graph_files = sorted(os.listdir(graph_dir))
        self.builder = PerfGraphBuilder()

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_file = self.graph_files[idx]
        prefix = os.path.splitext(graph_file)[0]

        graph_path = os.path.join(self.graph_dir, graph_file)
        label_path = os.path.join(self.label_dir, prefix + ".txt")

        nx_graph = load_compute_graph(graph_path)
        perfgraph = self.builder.build(nx_graph)

        with open(label_path, "r") as f:
            label_dict = eval(f.read())

        return perfgraph, label_dict
