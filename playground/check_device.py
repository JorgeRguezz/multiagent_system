from llama_cpp import Llama

model_name = "/home/gatv-projects2/Desktop/project/Qwen3-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
try:
    llm = Llama(
        model_path=model_name,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=True
    )
    print("Llama model initialized successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
