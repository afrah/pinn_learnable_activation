import torch

print("Version", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("NCCL Version:", torch.cuda.nccl.version())
