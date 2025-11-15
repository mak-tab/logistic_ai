from unsloth import FastLanguageModel
import torch
from peft import PeftModel, PeftConfig

# 1. Настройки
max_seq_length = 2048
dtype = None 
load_in_4bit = True 
adapter_path = "model_gguf" 

# 2. Загружаем БАЗОВУЮ модель (с явным управлением памятью)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пытаемся заставить ее поместиться, 
    # а главное - избежать ошибки "dispatched on the CPU"
    device_map = "auto", 
    trust_remote_code = True,
)

# 3. Загружаем конфигурацию LoRA из папки
try:
    peft_config = PeftConfig.from_pretrained(adapter_path, is_local=True)
except ValueError as e:
    # Здесь она найдет созданный тобой вручную adapter_config.json
    peft_config = PeftConfig.from_json_file(f"{adapter_path}/adapter_config.json")

# 4. Создаем PeftModel
model = PeftModel(model, peft_config, adapter_name="lora_adapter")

# 5. Сливаем LoRA-веса с базовой моделью (Merging)
print("Merging LoRA weights into base model...")
# Здесь произойдет слияние весов, которое освободит память LoRA-адаптера
model = model.merge_and_unload() 

# 6. Экспорт в GGUF (теперь с максимально чистым состоянием)
print("Starting GGUF conversion...")
model.save_pretrained_gguf(adapter_path, tokenizer, quantization_method = "q4_k_m")
print(f"Done! File saved in '{adapter_path}/unsloth.Q4_K_M.gguf' ✅")
