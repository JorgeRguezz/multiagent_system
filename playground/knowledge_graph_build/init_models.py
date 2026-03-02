import torch
import os
import re

class BaseModel:
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError

class LlamaModel(BaseModel):
    def __init__(self):
        print("Loading Llama 3.2 model...")
        from vllm import LLM, SamplingParams
        # Assuming Llama 3.2 3B path from context or standard location
        # The user didn't explicitly provide the path in the referenced files for Llama,
        # but _llm.py has "/home/gatv-projects/Desktop/project/llama-3.2-3B-Instruct"
        self.model_path = "/home/gatv-projects/Desktop/project/llama-3.2-3B-Instruct"
        self.llm = LLM(model=self.model_path, trust_remote_code=True, max_model_len=8192)
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        # Simple chat formatting
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        outputs = self.llm.generate([full_prompt], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

class DeepseekModel(BaseModel):
    def __init__(self):
        print("Loading Deepseek R1 model...")
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model="Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ",
            dtype="float16",
            trust_remote_code=True,
            max_model_len=4096,
            quantization="awq",
        )
        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=4096,
        )

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"<s><|begin_system|>\n{system_prompt or 'You are a helpful assistant.'}\n<|begin_user|>\n{prompt}\n<|begin_assistant|>\n"
        outputs = self.llm.generate([full_prompt], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        if "</think>" in generated_text:
            return re.split(r"</think>\s*", generated_text, maxsplit=1)[1].strip()
        return generated_text

class AprielModel(BaseModel):
    def __init__(self):
        print("Loading Apriel 1.6 model...")
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model_id = "ServiceNow-AI/Apriel-1.6-15b-Thinker"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"":0},
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"<s><|begin_system|>\n{system_prompt or 'You are a helpful assistant.'}\n<|begin_user|>\n{prompt}\n<|begin_assistant|>\n"
        
        inputs = self.processor(text=full_prompt, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        if "[BEGIN FINAL RESPONSE]" in output:
             return re.findall(r"\\[BEGIN FINAL RESPONSE\\](.*?)(?:<\|end\|>|$)", output, re.DOTALL)[0].strip()
        return output

class GPTModel(BaseModel):
    def __init__(self):
        print("Loading GPT-OSS 20B model...")
        from huggingface_hub import snapshot_download
        from llama_cpp import Llama

        model_id = "unsloth/gpt-oss-20b-GGUF"
        quant_file = "gpt-oss-20b-Q4_K_M.gguf"
        
        # Ensure model is downloaded
        model_path = snapshot_download(
            repo_id=model_id, 
            local_dir="./playground/gpt-oss-20b",
            allow_patterns=[quant_file]    
        )
        full_path = os.path.join(model_path, quant_file)
        
        self.llm = Llama(
            model_path=full_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=512,
            f16_kv=True,
            verbose=False
        )

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"<s><|begin_system|>\n{system_prompt or 'You are a helpful assistant.'}\n<|begin_user|>\n{prompt}\n<|begin_assistant|>\n"
        output = self.llm(
            full_prompt,
            max_tokens=12000,
            temperature=0.7,
            top_p=0.9,
            stop=["User:"]
        )
        clean_output = output['choices'][0]['text'].strip()
        
        if "<|message|>" in clean_output:
            return clean_output.split("<|message|>")[1].strip()
        elif "<|start|>assistant<|channel|>final|>" in clean_output:
            return clean_output.split("<|start|>assistant<|channel|>final|>")[1].strip()
        return clean_output

def load_model(model_name: str) -> BaseModel:
    model_name = model_name.lower()
    if model_name == "llama":
        return LlamaModel()
    elif model_name == "deepseek":
        return DeepseekModel()
    elif model_name == "apriel":
        return AprielModel()
    elif model_name == "gpt":
        return GPTModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: llama, deepseek, apriel, gpt")
