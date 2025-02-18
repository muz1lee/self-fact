# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import os 
import json
import fcntl

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments



def calculate_confidence_signals(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None):
        
    # 首先加载基础组件
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        
        # 加载数据集
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
        
        # 加载基座模型
        base_model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

        # 准备当前模型和参考模型
        if finetuning_args.use_ref_model:
            ref_model = base_model
            
            # 当前模型加载 adapter
            if finetuning_args.peft_path and finetuning_args.add_adapter:
                adapter_path = finetuning_args.peft_path
                adapter_config = os.path.join(adapter_path, "adapter_config.json")
                
                # 检查 adapter 配置文件是否存在
                if not os.path.exists(adapter_config):
                    print(f"Warning: Cannot find adapter config at {adapter_config}")
                    print("Using base model instead.")
                    model = base_model
                else:
                    from peft import PeftModel
                    print(f"Loading model adapter from {adapter_path}")
                    try:
                        model = PeftModel.from_pretrained(
                            base_model,
                            adapter_path,
                            is_trainable=False
                        )
                    except Exception as e:
                        print(f"Error loading adapter: {str(e)}")
                        print("Using base model instead.")
                        model = base_model
            else:
                model = base_model
        else:
            ref_model = None
            model = base_model

        # 准备数据整理器
        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,  # 现在 model 已经定义了
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

        # 更新参数
        training_args.remove_unused_columns = False
        # 初始化 trainer
        trainer = CustomDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **tokenizer_module,
        )

        # 获取训练集和验证集
        train_dataset = dataset_module['train_dataset']
        eval_dataset = dataset_module['eval_dataset']
        
        # 合并数据集
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_dataset,eval_dataset])
        
        total_samples = len(combined_dataset)
        print(f"Total samples for confidence calculation: {total_samples}")
        print(f"(Training: {len(train_dataset)}, Validation: {len(eval_dataset)})")
        
        # 在开始处理前，创建空的 JSON 文件
        output_file = f"{trainer.confidence_dir}/{trainer.save_confidence_name}.json"
        os.makedirs(trainer.confidence_dir, exist_ok=True)
        all_data = []
        
        model = trainer.model
        model.eval()
        
        # 重置计数器
        trainer.data_counter = 0
        
        print("\nCalculating confidence signals...")
        
        # 创建合并后数据集的 DataLoader
        combined_dataloader = DataLoader(
            combined_dataset,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        
        # 处理所有数据
        progress_bar = tqdm(total=total_samples, desc="Processing samples")
        for batch in combined_dataloader:
            batch = trainer._prepare_inputs(batch)
            with torch.no_grad():
                entries = trainer.get_confidence_signal(
                    model=model,
                    batch=batch
                )
                
                all_data.append(entries)
            
            progress_bar.update(1)
            
            # 定期保存数据到文件
            if len(all_data) % 100 == 0:  # 每处理100条数据保存一次
                save_data(all_data, output_file)
                print(f"Saved {len(all_data)} samples to file")
        
        progress_bar.close()
        
        # 最后保存所有数据
        save_data(all_data, output_file)
        
        print(f"\nProcessed and saved {len(all_data)} samples.")
        print(f"Results saved to: {output_file}")


def save_data(data, file_path):
    # 确保 data 是单个列表而不是列表的列表
    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]  # 展平嵌套列表
        
    with open(file_path, 'w', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, ensure_ascii=False, indent=4)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # 根据 get_confidence 参数执行不同的流程
    if finetuning_args.get_confidence:
        calculate_confidence_signals(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            callbacks=callbacks
        )
        
    else:
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

        # Create reference model
        if finetuning_args.use_ref_model:
            if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
                ref_model = model
            else:
                ref_model = create_ref_model(model_args, finetuning_args)
        else:
            ref_model = None

        # Update arguments
        training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

        # Initialize our Trainer
        trainer = CustomDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **tokenizer_module,
        )

        # Training
        if training_args.do_train:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            if finetuning_args.include_effective_tokens_per_second:
                train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                    dataset_module["train_dataset"], train_result.metrics, stage="rm"
                )

            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate(metric_key_prefix="eval")
            if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
                remove_keys = [key for key in metrics.keys() if "rewards" in key]
                for key in remove_keys:
                    metrics.pop(key)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Create model card
        create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)