import torch
import xgboost as xgb
import sklearn
import shap
import onnx

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA OK  : {torch.cuda.is_available()}")
print(f"GPU      : {device_name}")
print(f"XGBoost  : {xgb.__version__}")
print(f"sklearn  : {sklearn.__version__}")
print(f"SHAP     : {shap.__version__}")
print(f"ONNX     : {onnx.__version__}")
print(f"Device   : {device}")
