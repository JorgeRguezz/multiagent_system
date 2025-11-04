import torch
import platform
import sys

print(f"--- System Information ---")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print("")
print(f"--- PyTorch Information ---")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Torch built with CUDA: {torch.version.cuda}")
print("")

if torch.cuda.is_available():
    print(f"--- GPU Information ---")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        print(f"--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(i)}")
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"Total memory: {total_mem:.2f} GB")
        allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"Allocated memory: {allocated_mem:.2f} GB")
        print(f"Reserved memory (Cached): {reserved_mem:.2f} GB")
        free_mem = total_mem - reserved_mem
        print(f"Free memory: {free_mem:.2f} GB")
else:
    print("--- No GPU found ---")
