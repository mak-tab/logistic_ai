from unsloth import FastLanguageModel
import torch
import os

# 1. Настройки
max_seq_length = 1024 
dtype = None
load_in_4bit = True # Важно: загружаем в 4 бита, чтобы влезло в память

print("🔄 Загружаю обученные адаптеры из папки 'lora_adapters'...")

# 2. Загружаем модель СРАЗУ с твоими адаптерами
# Обрати внимание: model_name указывает на ПАПКУ, а не на HuggingFace
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_adapters", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
except OSError:
    print("❌ Ошибка: Папка 'lora_adapters' не найдена. Ты уверен, что обучение сохранилось?")
    exit()

print("💾 Начинаю конвертацию в GGUF (q4_k_m)...")
print("⚠️ Это может занять 5-10 минут и загрузить CPU на 100%. Не трогай комп.")

# 3. Конвертация
model.save_pretrained_gguf(
    "model_gguf", # Имя выходной папки/файла
    tokenizer,
    quantization_method = "q4_k_m"
)

print("✅ Готово! Файл должен лежать в папке 'model_gguf'")