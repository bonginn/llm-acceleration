"""
This script shows how we use LoRA to finetune 1B distilled model.
Estimated training time: ~1.5 hour per epoch on A100 40GB.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from utils.hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
from peft import PeftModel, LoraConfig, get_peft_model
from utils.trainer import CustomSFTTrainer

############## Set Up ##############
torch.manual_seed(0)
random.seed(0)
max_new_tokens = 256    # Number of new tokens to generate
device = 'cuda:0'
TEXT_COLUMN_NAME = "text"
MAX_TOKEN_LENGTH = 512
MAP_BATCH_SIZE = 4
DATALOADER_BATCH_SIZE = 2
DISABLE_CACHE_FOR_DEBUG = False # True


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=True,
)

print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

print("Loading datasets...")
try:
    train_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"Raw train dataset loaded with {len(train_dataset_raw)} samples.")
    print(f"Raw test dataset loaded with {len(test_dataset_raw)} samples.")
    if len(train_dataset_raw) > 0:
        print(f"First raw training sample: {train_dataset_raw[0]}")
        if TEXT_COLUMN_NAME not in train_dataset_raw.column_names:
            print(f"WARNING: Expected text column '{TEXT_COLUMN_NAME}' not found. Available columns: {train_dataset_raw.column_names}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

if torch.backends.mps.is_available():
    print("Using MPS backend for training.")
    device = "mps"
elif torch.cuda.is_available():
    print("Using CUDA backend for training.")
    device = "cuda"
else:
    print("Using CPU backend for training.")
    device = "cpu"

torch_dtype = torch.bfloat16

# Load Model and Merge Adapter
model_path = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = "bonginn/Llama-3.2-1B-Instruct-gkd-distilled-2000-Adapter"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map=device,
    use_cache=True,
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(model, adapter_path, low_cpu_mem_usage=True).cuda()
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.chat_template is None:
    print(f"WARNING: tokenizer.chat_template is None for {model_path}. Attempting to set a default or raise error.")

else:
    print(f"Tokenizer chat template found: {tokenizer.chat_template}")

try :
    model.to(device)
except :
    print("Techer Model already in GPU ? ")

def format_wikitext_for_gkd(examples):
    input_batch_size = len(examples["text"])
    #print(f"DEBUG: format_wikitext_for_gkd received batch with {input_batch_size} texts.")

    formatted_messages_batch = []
    for idx, text_content in enumerate(examples["text"]):
        if text_content is None:
            print(f"WARNING: Encountered None text_content at index {idx} in a batch.")

        formatted_messages_batch.append([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": str(text_content)}
        ])

    output_batch_size = len(formatted_messages_batch)
    #print(f"DEBUG: format_wikitext_for_gkd producing batch with {output_batch_size} messages.")

    if input_batch_size != output_batch_size:
        print(f"ERROR: Length mismatch! Input: {input_batch_size}, Output: {output_batch_size}")


    return {"messages": formatted_messages_batch}

train_dataset_formatted = train_dataset_raw.map(format_wikitext_for_gkd, batched=True)
test_dataset_formatted = test_dataset_raw.map(format_wikitext_for_gkd, batched=True)

lora_config = LoraConfig(
    r=64,                        # LoRA rank
    lora_alpha=128,               # scaling parameter
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
)

ft_args = TrainingArguments(
    output_dir="./finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=100,
    eval_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    remove_unused_columns=False,
)

trainer = CustomSFTTrainer(
    model=model,
    args=ft_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=test_dataset_formatted,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("Fine-tuning Model...")
trainer.train()