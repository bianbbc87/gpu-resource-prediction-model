import yaml
from google.cloud import storage

def load_config(path: str):
    """
    Load YAML config from local path or GCS (gs://).
    """
    if path.startswith("gs://"):
        # gs://bucket/path/to/file.yaml
        bucket_name, blob_path = path[5:].split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise FileNotFoundError(f"GCS config not found: {path}")

        content = blob.download_as_text()
        return yaml.safe_load(content)

    else:
        # local file
        with open(path, "r") as f:
            return yaml.safe_load(f)
