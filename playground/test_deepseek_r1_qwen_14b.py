from vllm import LLM, SamplingParams
import re

# 1) Load model with AWQ quantization
llm = LLM(
    model="Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ",
    dtype="float16",          # compute dtype
    trust_remote_code=True,
    max_model_len=4096,
    quantization="awq",       
)

# 2) Your context-generator prompt
video_game_name = "League of Legends"

prompt = f"""### INSTRUCTION
You are a Context Generator for an Entity Extraction AI.
Your goal is to provide a **concise, keyword-focused summary** of the video game "{video_game_name}".

The output will be used to help an AI understand proper nouns and specific terminology in text.
You must IGNORE release dates, developers, sales history, reviews, or graphical fidelity.

### CONTENT REQUIREMENTS
Focus ONLY on these three categories:
1.  **Setting & Premise:** The fictional world, time period, and main conflict.
2.  **Key Factions & Figures:** Major protagonists, antagonists, and groups (e.g., "The Empire", "Mario", "Ganon").
3.  **Unique Vocabulary:** Specific names for currency, magic, items, or mechanics (e.g., "Rupees", "Mana", "Drifting", "Fatality").

### FORMATTING RULES
- Keep the total output under 300 words.
- Use dense, informative sentences.
- Do not use bullet points; write as a cohesive summary paragraph.
- Do NOT explain your reasoning, just output the paragraph.

### EXAMPLE (Pac-Man)
Pac-Man is an arcade maze game set in a neon labyrinth. The protagonist, Pac-Man, must navigate the maze to eat dots and "Power Pellets" while avoiding four ghosts: Blinky, Pinky, Inky, and Clyde. Consuming a Power Pellet turns the ghosts blue, allowing Pac-Man to eat them for points. Fruit items occasionally appear as bonuses.

### INPUT
Target Game: {video_game_name}

### OUTPUT
"""

system_prompt = """
    You are a concise, domain-aware assistant.
    For this task: 
    - Answer directly without showing your reasoning steps.
    - Keep the output under 300 words.
    - Write single cohesive paragraphs without bullet points.
"""

full_prompt = f"""<s><|begin_system|>
{system_prompt}
<|begin_user|>
{prompt}
<|begin_assistant|>
"""

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=4096,
)

outputs = llm.generate([full_prompt], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text.strip()
print("="*40)
print("Generated Context Summary:")
print(generated_text)
print("="*40)

if "</think>" in generated_text:
    response = re.split(r"</think>\s*", generated_text, maxsplit=1)[1].strip()
else :
    response = "MALFORMED OUTPUT: Missing </think> tag."

print("Final Response:")
print(response)
print("="*40)

# llm.destroy_process_group()
