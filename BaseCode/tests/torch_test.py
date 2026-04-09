import torch, torchvision
print(f"\tPyTorch: {torch.__version__}")
print(f"\tTorchvision: {torchvision.__version__}")
print(f"\t¿CUDA disponible?: {torch.cuda.is_available()}")
print(f"\tVersión CUDA: {torch.version.cuda}")
