from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
import torch
import wandb

'''
#install the funtune nonsense
#import subprocess

#def install_package(command):
    #try:
        #subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print("Package installed successfully.")
    #except subprocess.CalledProcessError as e:
        #print("Failed to install package.")
        #print(e.output.decode())

#major_version, minor_version = torch.cuda.get_device_capability()

#if major_version >= 8:
    ## Use this for new GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)
    #install_package("pip install 'unsloth[colab-ampere] @ git+https://github.com/unslothai/unsloth.git'")
#else:
    # Use this for older GPUs (V100, Tesla T4, RTX 20xx)
    #install_package("pip install 'unsloth[colab] @ git+https://github.com/unslothai/unsloth.git'")

'''

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
# Initialize LoRa config
config = LoraConfig(
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                  "gate_proj", "up_proj", "down_proj",],
    target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized

    # use_gradient_checkpointing = True,
    # random_state = 3407,
    # use_rslora = False,  # We support rank stabilized LoRA
    # loftq_config = None, # And LoftQ
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

from unsloth.chat_templates import get_chat_template

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

from datasets import load_dataset
dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
       # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        #report_to="wandb",
        use_cache=False,
    ),
)

# train model
trainer.train()

wandb.finish()