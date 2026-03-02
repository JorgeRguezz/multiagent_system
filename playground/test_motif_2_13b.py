from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Motif-Technologies/Motif-2-12.7B-Instruct",
    trust_remote_code = True,
    _attn_implementation = "flash_attention_2",
    dtype = torch.bfloat16 # currently supports bf16 only, for efficiency
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "Motif-Technologies/Motif-2-12.7B-Instruct",
    trust_remote_code = True,
)

query = "What is the capital city of South Korea?"
input_ids = tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'you are an helpful assistant'},
        {'role': 'user', 'content': query},
    ],
    add_generation_prompt = True,
    enable_thinking = False, # or True
    return_tensors='pt',
).cuda()

output = model.generate(input_ids, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
output = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens = False)
print(output)
