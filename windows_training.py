import torch
import math
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_optimal_config():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found!")
    
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
    print(f"✅    - Gradient Checkpointing: {gradient_checkpointing}")

    return micro_batch, grad_accum, gradient_checkpointing

micro_batch, grad_accum, use_gc = get_optimal_config()

model_name = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.unk_token 

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        standard_convo = []
        for turn in convo:
            role = "user" if turn["from"] == "human" else "assistant"
            standard_convo.append({"role": role, "content": turn["value"]})
        
        text = tokenizer.apply_chat_template(
            standard_convo, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    return { "text" : texts }

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"✅    -    {len(train_dataset)} training \n✅    -    {len(eval_dataset)} evaluation.")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    packing=False,
    args=TrainingArguments(
        output_dir="outputs_windows",
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_gc,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=10,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    ),
)

print("✅ Training")
trainer.train()

print("✅ Saving")
trainer.model.save_pretrained("final_adapters_windows")
tokenizer.save_pretrained("final_adapters_windows")
print("✅ Done!")
