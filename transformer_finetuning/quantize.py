from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
import torch
import wandb

import os


# Import and configure pretrained model from HuggingFace
model_name = "meta-llama/Llama-2-7b-hf"
model_id = "leba01/Llama-2-7b-hf-8bit-quant"  #CHANGE THIS 

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right",)
tokenizer.pad_token = tokenizer.eos_token #LOOK INTO THIS


tokenizer.push_to_hub(model_id, use_temp_dir=True, repo_type="model")
model.push_to_hub(model_id, use_temp_dir=True, repo_type="model")

print("Finished uploading quantized model to HF")
