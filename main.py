import torch

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda count: {torch.cuda.device_count()}")
print(f"cuda name: {torch.cuda.get_device_name(0)}")


print("hello world.")