#!/bin/bash

pretrained=$1
peft=$2

lm_eval --model hf \
  --model_args pretrained="$pretrained",peft="$peft" \
  --tasks hellaswag, lambada_openai, arc_challenge, arc_easy, winogrande, piqa \
  --device cuda:0 \
  --batch_size 8