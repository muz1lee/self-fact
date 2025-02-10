---
library_name: peft
license: other
base_model: Meta-Llama-3-8B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: mix_0.125
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mix_0.125

This model is a fine-tuned version of [Meta-Llama-3-8B-Instruct](https://huggingface.co/Meta-Llama-3-8B-Instruct) on the dpo_en_demo dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6716
- Rewards/chosen: 0.0327
- Rewards/rejected: -0.0287
- Rewards/accuracies: 0.6667
- Rewards/margins: 0.0614
- Logps/chosen: -420.3072
- Logps/rejected: -395.7345
- Logits/chosen: -0.1932
- Logits/rejected: -0.1011
- Rewards/mix Margin: 0.0077

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.6.0+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3