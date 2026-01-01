# PerfGraph → PyTorch dataset

import os
import logging
from torch.utils.data import Dataset
from graph.extractor import load_compute_graph, PerfGraphBuilder
from google.cloud import storage


class PerfGraphDataset(Dataset):
    """
    Dataset of (PerfGraph, label)
    """

    def __init__(self, graph_dir, label_dir):
        self.graph_dir = graph_dir
        self.label_dir = label_dir
        
        if graph_dir.startswith("gs://"):
            logging.info(">>> Loading GCS file list")
            bucket_name, blob_path = graph_dir[5:].split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=blob_path + "/")
            self.graph_files = sorted([
                blob.name.split("/")[-1] 
                for blob in blobs 
                if blob.name.endswith('.pkl')
            ])
            logging.info(f">>> Found {len(self.graph_files)} graph files")
        else:
            self.graph_files = sorted(os.listdir(graph_dir))
            
        self.builder = PerfGraphBuilder()

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_file = self.graph_files[idx]
        prefix = os.path.splitext(graph_file)[0]

        if self.graph_dir.startswith("gs://"):
            # GCS에서 파일 로드
            graph_path = f"{self.graph_dir}/{graph_file}"
            label_path = f"{self.label_dir}/{prefix}.txt"
            
            # GCS에서 graph 파일 다운로드
            bucket_name, blob_path = graph_path[5:].split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            graph_blob = bucket.blob(blob_path)
            graph_content = graph_blob.download_as_bytes()
            
            # GCS에서 label 파일 다운로드
            bucket_name, blob_path = label_path[5:].split("/", 1)
            label_blob = bucket.blob(blob_path)
            label_content = label_blob.download_as_text()
            
            # 임시 파일로 저장 후 로드
            import tempfile
            import pickle
            
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(graph_content)
                tmp.flush()
                nx_graph = load_compute_graph(tmp.name)
            
            label_dict = eval(label_content)
        else:
            # 로컬 파일 로드
            graph_path = os.path.join(self.graph_dir, graph_file)
            label_path = os.path.join(self.label_dir, prefix + ".txt")

            nx_graph = load_compute_graph(graph_path)

            with open(label_path, "r") as f:
                label_dict = eval(f.read())

        perfgraph = self.builder.build(nx_graph)
        return perfgraph, label_dict
