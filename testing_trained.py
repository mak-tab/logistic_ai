from unsloth import FastLanguageModel
import torch

# 1. Загружаем твою ТОЛЬКО ЧТО обученную модель
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Папка, куда сохранился результат (проверь имя папки в training.py)
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. Тестовый "грязный" пост (которого не было в обучении!)
test_post = """
🔥🔥🔥 СРОЧНО!
Ташкент - Бухара
Нужен реф, 20 тонн.
Груз: мороженое.
Оплата 3.000.000 сум нал.
Звонить: +998 90 123 45 67
"""

# 3. Формируем промпт (ТОЧНО ТАКОЙ ЖЕ, как при обучении)
prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a logistics AI. Extract shipments from the text into a JSON list.

### Input:
{test_post}

### Response:
"""

# 4. Запускаем генерацию
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens = 512, 
    use_cache = True
)

# 5. Декодируем ответ
result = tokenizer.batch_decode(outputs)
print("\n=== РЕЗУЛЬТАТ ===\n")
print(result[0].split("### Response:")[-1].replace("<|endoftext|>", ""))