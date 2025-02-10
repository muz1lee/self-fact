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


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 使用 rm 阶段加载数据集
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
    training_args.remove_unused_columns = False

    # Initialize trainer
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

    if finetuning_args.get_confidence:
        train_dataset = dataset_module['train_dataset']
        print(f"Total samples for confidence calculation: {len(train_dataset)}")
        
        # 使用 SequentialSampler 确保顺序性和不重复
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        
        # 修改 DataLoader 配置
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            sampler=sampler,  # 使用顺序采样器
            shuffle=False,
            drop_last=False,
            num_workers=0,    # 禁用多进程加载
            pin_memory=False  # 禁用 pin_memory
        )
        
        model = trainer.model
        model.eval()
        
        # 重置计数器
        trainer.data_counter = 0
        
        # 确定当前任务类型
        task = finetuning_args.confidence_type
        print(f"\nRunning {task} logps calculation...")
        
        # 使用 tqdm 来显示进度
        for batch in tqdm(train_dataloader, desc=f"Processing {task} logps"):
            batch = trainer._prepare_inputs(batch)
            with torch.no_grad():
                trainer.get_batch_logps(
                    model=model,
                    batch=batch,
                    task=task,
                    train_eval="train"
                )
        
        print(f"\nProcessed {trainer.data_counter} samples for {task} model.")
        print(f"Results saved to: {trainer.confidence_dir}/{task}_logps.json")

    else:
        # 正常的训练流程
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
    # create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
