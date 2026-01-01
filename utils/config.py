import yaml
import logging
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import Forbidden, PermissionDenied

def load_config(path: str):
    logging.info(f">>> load_config called with path: {path}")

    if path.startswith("gs://"):
        bucket_name, blob_path = path[5:].split("/", 1)

        try:
            logging.info(">>> Creating GCS client")
            client = storage.Client()

            logging.info(f">>> Accessing bucket: {bucket_name}")
            bucket = client.bucket(bucket_name)

            logging.info(f">>> Accessing blob: {blob_path}")
            blob = bucket.blob(blob_path)

            logging.info(">>> Checking blob exists()")
            exists = blob.exists()

            if not exists:
                raise FileNotFoundError(f"GCS config not found: {path}")

            logging.info(">>> Downloading config")
            content = blob.download_as_text()

            logging.info(">>> Config downloaded successfully")
            return yaml.safe_load(content)

        except DefaultCredentialsError as e:
            logging.error("GCS AUTH ERROR: Default credentials not found")
            logging.exception(e)
            raise

        except (Forbidden, PermissionDenied) as e:
            logging.error("GCS AUTH ERROR: Permission denied")
            logging.exception(e)
            raise

        except Exception as e:
            logging.error("UNEXPECTED ERROR while loading config from GCS")
            logging.exception(e)
            raise

    else:
        logging.info(">>> Loading local config file")
        with open(path, "r") as f:
            return yaml.safe_load(f)