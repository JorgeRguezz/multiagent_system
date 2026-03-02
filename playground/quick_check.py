import llama_cpp
print(f"llama_cpp version: {llama_cpp.__version__}")
print("GPU offload supported:", llama_cpp.llama_supports_gpu_offload())