## Описание

Проект использует:
- **Pyrogram** для работы с Telegram API
- **LLaMA-based LM** (Gemma-3.1B) для парсинга текста
- **Fine-tuning с LoRA** (Unsloth + Phi-3) для повышения точности

Бот мониторит указанные чаты Telegram, извлекает информацию о грузовых перевозках и преобразует "грязные" текстовые сообщения в структурированный JSON формат.

## Функционал

### Извлекаемые данные:
- **from** — город отправления
- **to** — город назначения  
- **cargo** — тип груза и вес
- **vehicle** — тип транспорта (тент, реф, изотерма и т.д.)
- **price** — стоимость перевозки
- **phone_number** — контактный номер

### Пример входных данных:
```
Ташкент - Самарканд
22 тонны, масло в коробках
Нужен рефрижератор
3.700.000 сум наличными
```

### Пример выходных данных:
```json
[{
  "from": "Ташкент",
  "to": "Самарканд",
  "cargo": "Масло в коробках, 22 тонны",
  "vehicle": "Рефрижератор",
  "price": "3.700.000 сум наличными",
  "phone_number": null
}]
```
## Старт

### 1. Установка зависимостей

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
# для запуска
pip install -r requirements.txt
```
```bash
# ДЛя обучения модели
pip install -r requirements-dev.txt
```

### 2. Конфигурация

Создайте/обновите файл `.env`:

```env
API_ID=12312312
API_HASH="your_api_hash_here"
PHONE="+998xxxxxxxxx"
CHAT_IDS="-100xxx,-100xxx,-100xxx"
```

- **Получение API_ID и API_HASH:**  
- Зарегистрируйтесь на https://my.telegram.org → API Development Tools

### 3. Загрузка модели

Скачайте предобученную модель:
```bash
# Модель Gemma-3.1B квантизованная (Q4)
wget https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf/
```

### 4. Запуск бота

```bash
python bot.py
```

# Но промптинг оказался слабым для подобных моделей, поэтому решил обучить модель для более точности

## Обучение собственной модели

### Подготовка датасета

Датасет должен быть в формате JSONL с диалогами:

```jsonl
{"conversations": [{"from": "human", "value": "TEXT..."}, {"from": "gpt", "value": "[JSON...]"}]}
```

Смотрите в `dataset.jsonl`.

### Обучение (Fine-tuning)

```bash
python training.py
```

**Параметры обучения:**
- Модель: `unsloth/Phi-3-mini-4k-instruct`
- LoRA rank: 16
- Батч: 2, градиент аккумуляция: 4
- Шаги: 60, learning_rate: 2e-4
- Оптимизатор: adamw_8bit

### Конвертация в GGUF

```bash
python gguf.py
```

Модель будет сохранена в `model_gguf/unsloth.Q4_K_M.gguf`.

### Тестирование

```bash
python testing_trained.py
```

## Как работает парсер ???

`parser.py` использует следующий подход:

1. **Очистка текста**: удаление эмодзи, нормализация пробелов
2. **Промпт для LM**: система передаёт инструкции модели
3. **Генерация**: модель возвращает JSON
4. **Валидация**: проверка структуры и нормализация данных

## API Telegram

`bot.py` использует **Pyrogram** для:
- Получения истории чатов
- Мониторинга новых сообщений
- Обработки текста и медиа

## Примеры +-

### Запуск с одним чатом

```python
# В .env
CHAT_IDS="-1001331922767"
```

### Обработка файла с текстами

```python
from parser import parsing

with open('logistics_texts.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
orders = parsing(text)
for order in orders:
    print(json.dumps(order, ensure_ascii=False, indent=2))
```

## Требования

- **Python**: 3.10+
- **CUDA**: 12.1+ (для обучения)
- **RAM**: 8+ ГБ (для работы бота), 24+ ГБ (для обучения)
- **GPU**: рекомендуется NVIDIA с CUDA поддержкой и VRAM 8+

## Основные библиотеки

| Библиотека | Назначение |
|-----------|-----------|
| pyrogram | Telegram API |
| llama-cpp-python | Работа с GGUF моделями |
| torch | Deep Learning (обучение) |
| unsloth | Оптимизация обучения LoRA |
| peft | Parameter-Efficient Fine-tuning |
| datasets | Загрузка датасетов |

## Решение проблем

### Ошибка: "pip install -r requirements-dev.txt"
```bash
pip install --upgrade pip setuptools wheel
```
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
```bash
pip install trl peft accelerate datasets bitsandbytes
```

### Telegram: "AUTH_KEY_INVALID"
```bash
rm telegram_logistic.session
python bot.py
```

### OOM при обучении
```python
per_device_train_batch_size = 1  # Вместо 2
```