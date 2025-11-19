from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import math
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import gc
import time

def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 4, True
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"✅   -   GPU: {gpu_name} | VRAM: {total_vram_gb:.2f} GB")

    TARGET_GLOBAL_BATCH_SIZE = 32 

    if total_vram_gb <= 6.5:
        micro_batch = 2
        gradient_checkpointing = True
    elif total_vram_gb <= 12.5:
        micro_batch = 4
        gradient_checkpointing = True
    elif total_vram_gb <= 16.5:
        micro_batch = 8
        gradient_checkpointing = True
    elif total_vram_gb <= 24.5:
        micro_batch = 16 
        gradient_checkpointing = False 
    else:
        micro_batch = 32
        gradient_checkpointing = False

    grad_accum = math.ceil(TARGET_GLOBAL_BATCH_SIZE / micro_batch)

    print(f"✅ Config:")
    print(f"✅    - Micro Batch Size: {micro_batch}")
    print(f"✅    - Gradient Accumulation: {grad_accum}")
    print(f"✅    - Effective Batch: {micro_batch * grad_accum}")
    print(f"✅    - Gradient Checkpointing: {gradient_checkpointing}")

    return micro_batch, grad_accum, gradient_checkpointing

micro_batch, grad_accum, use_gc = get_optimal_config()

max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0.05, 
    bias = "none", 
    use_gradient_checkpointing = use_gc, 
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3",
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"    -    {len(train_dataset)} training \n\n    -    {len(eval_dataset)} evaluation.")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = micro_batch, 
        gradient_accumulation_steps = grad_accum,
        
        warmup_ratio = 0.1,
        num_train_epochs = 10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        eval_strategy = "steps",
        eval_steps = 5,
        save_strategy = "steps",
        save_steps = 5,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        report_to = "none",
    ),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✅   -   Training begins!")
trainer.train()
print("✅   -   Training finished!")

model.save_pretrained("lora_adapters")
tokenizer.save_pretrained("lora_adapters")

print("✅   -   Cleaning up memory for GGUF conversion")

del trainer
gc.collect()
torch.cuda.empty_cache()

time.sleep(10) 

print("✅   -   Memory released. Beginning GGUF conversion")

try:
    model.save_pretrained_gguf(
        "model_gguf", 
        tokenizer, 
        quantization_method = "q4_k_m"
    )
    print("✅   -   Done!")
except Exception as e:
    print(f"✅   -   GGUF conversion failed: {e}")
    print("✅   -   Don't worry, adapters are saved in 'lora_adapters'. Use manual conversion if needed.")
