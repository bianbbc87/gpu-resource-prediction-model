# ONNX 모델 로딩

import onnx

def load_onnx_model(onnx_path: str):
    """
    Load ONNX model.
    이후 graph_parser에서 사용
    """
    return onnx.load(onnx_path)
