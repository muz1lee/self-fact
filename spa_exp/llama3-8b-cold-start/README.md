---
library_name: peft
license: other
base_model: /mnt/sda/llmfact/models/Meta-Llama-3-8B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: llama3-8b-cold-start
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama3-8b-cold-start

This model is a fine-tuned version of [/mnt/sda/llmfact/models/Meta-Llama-3-8B-Instruct](https://huggingface.co//mnt/sda/llmfact/models/Meta-Llama-3-8B-Instruct) on the biography_mix dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6910
- Rewards/chosen: 0.0180
- Rewards/rejected: 0.0133
- Rewards/accuracies: 0.5385
- Rewards/margins: 0.0048
- Rewards/mix Margin: 0.0035
- Logps/chosen: -166.4122
- Logps/rejected: -177.1431
- Logits/chosen: -0.0886
- Logits/rejected: -0.0916

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
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3