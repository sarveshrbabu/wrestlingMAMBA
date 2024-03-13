import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
from unsloth.chat_templates import get_chat_template
import wandb
import os

#get wandb stuff
os.environ["WANDB_PROJECT"] = "wrestlingMamba"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

wandb.login()


# Load the SlimOrca-Dedup dataset
dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")

# Split the 'train' split into training and validation sets
train_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=42)

# Load the Pythia 160M tokenizer and model
model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create LoRA model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["from"] + " " + examples["value"], truncation=True, padding="max_length", max_length=512)


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

#dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split = "train")
#dataset = dataset.map(formatting_prompts_func, batched = True,)

tokenized_train_dataset = train_data.map(formatting_prompts_func, batched=True, remove_columns=["from", "value"])
tokenized_validation_dataset = validation_data.map(formatting_prompts_func, batched=True, remove_columns=["from", "value"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./pythia_slim_orca_dedup_sft_lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
)

# Create the SFT Trainer with LoRA model
sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# Train the model using SFT with LoRA
sft_trainer.train()

# Save the fine-tuned LoRA model
model.save_pretrained("./pythia_slim_orca_dedup_sft_lora_model")

wandb.finish()