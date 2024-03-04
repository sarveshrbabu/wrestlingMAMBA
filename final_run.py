import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Dataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from accelerate import Accelerator
from collections import namedtuple

# Monkey patch the forward method
def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels=None):
    """
    Customize the forward pass to handle the loss computation directly within the model for fine-tuning.
    """
    from torch.nn.functional import cross_entropy

    # Compute the hidden states from the backbone
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)

    if labels is not None:
        # Prepare logits and labels for loss computation
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss,)
    else:
        # Wrap logits in a namedtuple if no labels are provided
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

# Apply the monkey patch
MambaLMHeadModel.forward = forward_with_loss

# Configuration
modelpath = "state-spaces/mamba-1.4b"
bs = 4  # batch size
ga_steps = 1  # gradient accumulation steps
epochs = 4
lr = 0.00005
output_dir = "./out"

# Load the model and tokenizer
model = MambaLMHeadModel.from_pretrained(modelpath, dtype=torch.bfloat16, device="cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Add special tokens and resize embeddings
special_tokens_dict = {'additional_special_tokens': ['<system>', '<human>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Load and preprocess the dataset
json_file_path = 'path/to/your/dataset.json'  # Adjust the path to your dataset
raw_dataset = load_dataset('json', data_files=json_file_path)
def preprocess_conversations(example):
    # Concatenate conversation turns with speaker tags
    formatted_text = " ".join([f"<{conv['from']}> {conv['value']}" for conv in example['conversations']])
    return {"text": formatted_text}
dataset = raw_dataset.map(preprocess_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare for training with Accelerator
accelerator = Accelerator()
model, tokenized_datasets["train"] = accelerator.prepare(model, tokenized_datasets["train"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=bs,
    gradient_accumulation_steps=ga_steps,
    learning_rate=lr,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,  # Set to True if you want to push to the Hugging Face Hub
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['input_ids'] for f in data])},  # Labels are input_ids for LM
)

# Start training
trainer.train()
