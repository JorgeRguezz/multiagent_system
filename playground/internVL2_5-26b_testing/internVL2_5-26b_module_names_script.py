import torch
import accelerate

# Disable any accelerate big-model context that might be active
accelerate.init_empty_weights.__wrapped__ = None  # no-op safety

# Read model structure directly from config instead of instantiating the model
from transformers import AutoConfig

model_name = "OpenGVLab/InternVL2_5-26B"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Print the language model config (shows layer count)
print("=== Vision config ===")
print(config.vision_config)

print("\n=== Language model config ===")
print(config.llm_config)

print("\n=== Top-level config keys ===")
print(config.to_dict().keys())
 