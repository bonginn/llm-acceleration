"""
Example script for LLaMA-3.2 distillation using GKD.
Estimated training time: ~8-10 hours per epoch on A100 40GB.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling ,BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
from datasets import load_dataset
from trl import GKDConfig, GKDTrainer
from utils.callback import RealTimeLossCallback
from utils.trainer import CustomGKDTrainer

temperature = 10.0
teacher_lora_dir = "yutaliu/Llama-3.2-3B-wiki-Lora"
teacher_model_name = "meta-llama/Llama-3.2-3B-Instruct"
student_model_name = "meta-llama/Llama-3.2-1B-Instruct"

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=True,
)

print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

TEXT_COLUMN_NAME = "text"
MAX_TOKEN_LENGTH = 512
MAP_BATCH_SIZE = 4
DATALOADER_BATCH_SIZE = 2
DISABLE_CACHE_FOR_DEBUG = False # True

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

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    attn_implementation="sdpa",  # Flash Attention 2.0
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    use_cache=False,
    trust_remote_code=True
)

teacher_model = PeftModel.from_pretrained(
    teacher_model,
   teacher_lora_dir,  # LoRA checkpoint 路徑
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
)

student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    #quantization_config=bnb_config,  # training
    device_map="auto",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    use_cache=False,
    trust_remote_code=True
)

student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()

print("8 bit model loader")

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.chat_template is None:
    print(f"WARNING: tokenizer.chat_template is None for {teacher_model_name}. Attempting to set a default or raise error.")
else:
    print(f"Tokenizer chat template found: {tokenizer.chat_template}")

try :
    teacher_model.to(device)
except :
    print("Techer Model already in GPU ? ")

try :
    student_model.to(device)
except :
    print("Techer Model already in GPU ? ")

from trl import GKDConfig, GKDTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

output_dir = f"./llama3.2-1B-gkd-distilled-temp"+str(int(temperature))

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

training_args = GKDConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-6,
    num_train_epochs=1,
    warmup_steps=100,
    max_length=512,
    save_steps=200,
    eval_steps=200,
    save_total_limit=50,
    eval_strategy="steps",
    logging_steps=1,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    temperature=temperature,
    torch_empty_cache_steps=10,
    seed=42,
    bf16=True, #if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else False,
    fp16=False, #if torch.backends.mps.is_available() else (True if torch.cuda.is_available() else False),
    dataloader_num_workers=0,
)

num_epochs = 1
total_steps = len(train_dataset_formatted) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * num_epochs

from transformers import get_linear_schedule_with_warmup
if torch.cuda.is_available():
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        student_model.parameters(),
        lr=1e-6,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
else:
    #  mps
    from torch.optim import AdamW
    optimizer = AdamW(
        student_model.parameters(),
        lr=1e-6,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,
    num_training_steps=total_steps
)

trainer = CustomGKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset_formatted,
    eval_dataset=test_dataset_formatted,
    optimizers=(optimizer,scheduler),
    callbacks=[RealTimeLossCallback()],
)

try:
    trainer.train()
    print("\n Training completed successfully! \n")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model save: {output_dir}")

except Exception as e:
    print(f"Training error: {e}")