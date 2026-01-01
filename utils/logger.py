import logging

def build_logger(name):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
