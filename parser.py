import json
import re
import emoji
from llama_cpp import Llama

MODEL_PATH = "gemma-3-1b-pt-q4_0.gguf" 

def load_model():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=0,
        verbose=False,
        n_gpu_layers=-1
    )
    return llm

llm_instance = load_model()

def extract_json(text: str) -> str:
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return f"[{text[start:end+1]}]"
        
    return "[]"

def parsing(text: str) -> list[dict]:
    if not text:
        return []
        
    clean_text = emoji.replace_emoji(text, replace='')
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    system_prompt = """You are a simple parser for cargo transportation requests.

Your task: extract data ONLY if the text clearly contains:
- two cities (origin and destination),
- cargo or vehicle information,
- a phone number.

If the text does NOT look like a cargo transportation request, return:
[]

If the text DOES contain cargo-transport information, return ONLY this JSON structure:

[
  {
    "from": "",
    "to": "",
    "cargo": "",
    "vehicle": "",
    "phone_number": ""
  }
]

Extraction rules:
- Cities are words that look like real cities or are written in ALL CAPS.
- If you cannot identify a city, leave the field empty.
- If cargo info is missing, leave "".
- If vehicle info is missing, leave "".
- Extract a phone number from the text.
- DO NOT create multiple objects. Always return exactly ONE object or an empty array [].
- DO NOT guess, DO NOT invent missing information.
- Output ONLY valid JSON. No explanations.

Now, parse the following text. Return only JSON.
"""
    
    final_prompt = (
        f"<start_of_turn>user\n"
        f"{system_prompt}\n\n"
        f"Text: '''{clean_text}'''\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    response = llm_instance.create_completion(
        prompt=final_prompt,
        temperature=0.0,
        max_tokens=4096,
        stop=["<end_of_turn>"]
    )

    raw_output = response['choices'][0]['text']
        
    json_str = extract_json(raw_output)
        
    if json_str == "[]": return []

    parsed_data = json.loads(json_str)

    if isinstance(parsed_data, dict): parsed_data = [parsed_data]

    normalized_data = []
    required_keys = ["from", "to", "cargo", "vehicle", "price", "phone_number"]
        
    for item in parsed_data: 
        if not isinstance(item, dict): 
            continue
            
        normalized_item = {key: item.get(key) for key in required_keys}
            
        if normalized_item.get('from') or normalized_item.get('to') or normalized_item.get('phone_number'):
            normalized_data.append(normalized_item)
            
    return normalized_data
