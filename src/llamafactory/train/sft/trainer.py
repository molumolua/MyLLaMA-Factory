# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from datasets import load_dataset
from ..math_eval import process_reject_sample
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from ...extras.logging import get_logger
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

def get_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)  # 通过 DistributedSampler 将数据集分配到每张卡上
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4)
    return dataloader

logger = logging.get_logger(__name__)



class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    # @override
    # def evaluate(
    #     self,
    #     eval_dataset: Optional[Dataset] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    #     **gen_kwargs,
    # ) -> Dict[str, float]:
    #     # 初始化stop_words，dataset和model
    #     logger.info("Start Eval Math.")
    #     eval_dataset=load_dataset("HuggingFaceH4/MATH-500",split="test")
    #     model =self._wrap_model(self.model,training=False,dataloader=None)
    #     logger.info("Load Math Data Successful.")
    #     #处理dataset和param
    #     input_texts = [
    #                 self.tokenizer.apply_chat_template(
    #                         [    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}"}
    #                             ,{"role": "user", "content": problem['problem']}],
    #                         tokenize=False,
    #                         add_generation_prompt=True,
    #                 )
    #                 for problem in eval_dataset
    #     ]
    #     logger.info(input_texts[0])
    #     # sampling_params = SamplingParams(
    #     #     max_tokens=32678,
    #     #     temperature=0,
    #     #     stop=stop_words,
    #     #     n=1
    #     # )
    #     # input_texts = [
    #     #             self.tokenizer.apply_chat_template(
    #     #                     [    {"role": "user", "content": problem['problem']+"\nPlease reason step by step, and put your final answer within \\boxed{{}}"}],
    #     #                     tokenize=False,
    #     #                     add_generation_prompt=True,
    #     #             )
    #     #             for problem in eval_dataset
    #     # ]
    #     generated_responses = model.generate(input_texts, max_new_tokens=32678,
    #         temperature=0,
    #         eos_token_id=self.tokenizer.eos_token_id)
    #     generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]
    #     score=sum([
    #         process_reject_sample(problem,'solution',response,logger) for problem,response in zip(eval_dataset,generated_responses)
    #     ])/len(eval_dataset)*100
    #     return {f"{metric_key_prefix}_math_score":score}
    
    @override
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:
        logger.info_rank0("Start Eval Math.")
        
        eval_dataset = load_dataset("json", data_files="./data/test_50.json", split="train")
        
        logger.info_rank0(f"Load Math Data Success. Len {len(eval_dataset)}.")
        
        # Initialize distributed training if necessary
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Use DistributedSampler for multi-card setup, or no sampler for single-card
        batch_size = 4
        sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank) if dist.is_initialized() else None
        dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            pin_memory=True,       # 加速数据传输
            num_workers=4          # 根据实际情况调整进程数
        )
        
        # 设置 tokenizer 为左侧填充，解决 decoder-only 模型的右填充问题
        self.tokenizer.padding_side = "left"
        # 如果 pad_token_id 未设置，则尝试使用 eos_token_id 或 eos_token 对应的 id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        logger.info_rank0("Model Wrap Success.")

        generated_responses = []

        for i, batch in enumerate(tqdm(dataloader, desc="Processing Batches", ncols=100, unit="batch")):
            device = torch.device(f"cuda:{rank}" if dist.is_initialized() else "cuda:0")
            logger.info(f"Processing Batch {i} in {device}")
            with torch.no_grad():  # 禁用梯度计算以降低GPU占用
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    problems = batch['problem']
                    input_texts = [
                        self.tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}"},
                                {"role": "user", "content": problem}
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for problem in problems
                    ]
                    model_inputs = self.tokenizer(input_texts, return_tensors="pt",padding=True).to(device)

                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=2048
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    batch_decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_responses.extend(batch_decoded)

        # Gather results from all processes if distributed
        all_generated_responses = None
        if rank == 0:
            all_generated_responses = [generated_responses]
        else:
            all_generated_responses = dist.gather(generated_responses, dst=0)

        if rank == 0:
            all_generated_responses = sum(all_generated_responses, [])
            score = sum([
                process_reject_sample(problem, 'solution', response, logger, timeout=0)
                for problem, response in zip(eval_dataset, all_generated_responses)
            ]) / len(eval_dataset) * 100

            return {
                "steps": self.state.global_step,
                f"{metric_key_prefix}_math_score": score
            }
    @override   
    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        # eval MATH
        metrics_file = os.path.join(self.args.output_dir, "eval_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []  # Initialize an empty list if the file doesn't exist

        # Append the new metrics to the list
        all_metrics.append(metrics)

        # Save the updated metrics list back to the JSON file
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        return metrics
    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
