import time
import json
from pyrogram import Client
from pyrogram.errors import FloodWait

from parser import parsing

import os
from dotenv import load_dotenv

load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')
CHAT_IDS_STR = os.getenv('CHAT_IDS')
TARGET_CHATS = [int(id.strip()) for id in CHAT_IDS_STR.split(',') if id.strip()]
POLL_INTERVAL_SECONDS = 1

app = Client("telegram_logistic", api_id=API_ID, api_hash=API_HASH, phone_number=PHONE)

def get_initial_last_id(client: Client, chat_id: int) -> int:
    try:
        history_generator = client.get_chat_history(chat_id, limit=1)
        last_message_list = list(history_generator)
        
        if last_message_list:
            last_id = last_message_list[0].id
        else:
            last_id = 0
    except Exception as e:
        last_id = 0
    return last_id

def process_message(msg):
    print(f"\n    -    Чат: {msg.chat.id} | Сообщение ID: {msg.id}")
    message_text = msg.text or ""
    print("\n================================\n"
             "--------- ORIGINAL ---------\n"
             f"{message_text}\n"
             "------------------------------")
    
    parsed_orders_list = parsing(message_text)
    
    print(f"---------- JSON ----------")
    
    if not parsed_orders_list:
        return

    for i, order_data in enumerate(parsed_orders_list):
        json_output = json.dumps(order_data, indent=2, ensure_ascii=False)
        print(f"--- [ ЗАКАЗ {i+1} ] ---\n{json_output}")
        
    print("================================")

def run_poller():
    with app:
        chat_states = {}
        
        print("[Инициализация чатов...]")
        for chat_id in TARGET_CHATS:
            last_id = get_initial_last_id(app, chat_id)
            chat_states[chat_id] = last_id
            print(f"    -    Чат {chat_id} подключен. Последний ID: {last_id}")
        
        print("[Бот запущен]")
        
        while True:
            for chat_id in TARGET_CHATS:
                current_last_id = chat_states[chat_id]
                
                try:
                    messages_generator = app.get_chat_history(chat_id, limit=20)
                    messages_list = list(messages_generator)
                    
                    new_messages = [msg for msg in messages_list if msg.id > current_last_id]
                    
                    if new_messages:
                        for msg in reversed(new_messages):
                            process_message(msg)
                            chat_states[chat_id] = msg.id
                            
                except Exception as e:
                    print(f"Ошибка при чтении чата {chat_id}: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_poller()
    