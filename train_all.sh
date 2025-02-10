#!/bin/bash

# 激活环境
conda activate llamafactory

base_model_path=/mnt/sda/llmfact/LLaMA-Factory/Meta-Llama-3-8B-Instruct

for mixing in 1.0 0.5 0.25 0.125
do
    output_dir=/mnt/sda/llmfact/LLaMA-Factory/spa_exp/llama3-8b/distill/mix_${mixing}

    CUDA_VISIBLE_DEVICES=3 llamafactory-cli train \
        examples/train_lora/llama3_lora_dpo.yaml \
        --output_dir=${output_dir} \
        --pref_loss=${distill} \
        --decouple_denoising=${mixing} \
        --confidence_dir=${output_dir} \
        --model_name_or_path=${base_model_path}
done
