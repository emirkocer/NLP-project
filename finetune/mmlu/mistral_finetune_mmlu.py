from unsloth import FastLanguageModel
import torch
import sys
import wandb
from dotenv import load_dotenv
import os
load_dotenv()
wandb_api_key = os.getenv('HUGGINGFACE_API_KEY')
wandb.login(key=wandb_api_key)
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Prepare the train.jsonl file for fine-tuning
from datasets import load_dataset, Dataset
import json

EOS_TOKEN = tokenizer.eos_token  # Assuming 'tokenizer' has been defined and loaded

mmlu_prompt = """Answer the following multiple-choice question about chemistry or physics or biology
by selecting the correct option: 'A', 'B', 'C', or 'D'. Only give the
correct option as the answer without reasoning.\n
Question:
{}

Options:
{}

Answer:
{}
"""
def formatting_prompts_func(batch):
    # Extract lists of problems and solutions from the nested structure
    questions = [item['question'] for item in batch['data']]
    choices = [item['choices'] for item in batch['data']]
    answers = [item['answer'] for item in batch['data']]
    texts = []
    for question, choice, answer in zip(questions, choices, answers):
        choices_text = "\n".join([f"{chr(65+i)}: {opt}" for i,opt in enumerate(choice)])
        answer_letter = chr(65 + answer)
        text = mmlu_prompt.format(question, choices_text, answer_letter) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def prepare_dataset_text(questions, answers, choices):
    texts = []
    for question, choice, answer in zip(questions, choices, answers):
        text = f"### Question:\n{question}\n{choices_text}:\n ### Answer:\n{answer_letter}\n" + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def load_jsonl_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file.readlines()]
    return Dataset.from_dict({'data': data})

# Load and prepare your dataset
aux_train = load_dataset("cais/mmlu", "auxiliary_train")
train_dataset = aux_train["train"]

with open('mmlu_train.jsonl', 'w') as f:
    for item in train_dataset:
        item = item["train"]
        f.write(json.dumps({'question': item['question'], 'choices': item['choices'], 'answer': item['answer']}) + '\n')

mmlu_train_dataset = load_jsonl_to_dataset('mmlu_train.jsonl')
dataset = mmlu_train_dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 32, # adjusted based on the memory usage
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = 1,
        max_steps = 240,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb"
    ),
)

# Show current GPU stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")







