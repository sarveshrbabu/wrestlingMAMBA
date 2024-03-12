from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
import evaluate
import torch


# Import and configure pretrained model from HuggingFace
model_name = "EleutherAI/pythia-2.8b"
bnb_config = BitsAndBytesConfig(
load_in_8bit=True,
bnb_8bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
print(model)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right",)
tokenizer.pad_token = tokenizer.eos_token #LOOK INTO THIS

# Prepare PEFT model for training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Initialize LoRa config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Import and format dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
dataset = dataset.map(lambda x: tokenizer(x["text"], return_tensors="pt"), batched=True) #EDIT THIS FOR SURE

# Define hyperparameters
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
learning_rate = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir= "output",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)

# configure trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train_sft"],
    eval_dataset=dataset["test_sft"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# train model
trainer.train()