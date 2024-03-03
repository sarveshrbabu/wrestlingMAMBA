#!/bin/bash

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the preprocessing script
python processing.py

# Run the training script with specified parameters
python train_mamba.py --model state-spaces/mamba-2.8b --tokenizer EleutherAI/gpt-neox-20b --learning_rate 5e-5 --batch_size 1 --gradient_accumulation_steps 4 --optim paged_adamw_8bit --num_epochs 3

#instructions 
#run this in bash 
#chmod +x run.sh
#./run.sh
