# 0. imports
import os
import random
import warnings
import pprint
from dataclasses import dataclass, field
from collections import deque
from typing import Literal
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tyro
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.utils.data.dataloader import DataLoader

from trl import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding


@dataclass
class ScriptArguments:
    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )

    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    dataset_path: Optional[str] = field(
        default="tatsu-lab/alpaca_farm", metadata={"help": "the dataset name"}
    )
    dataset_name: Literal[
        "alpaca_human_preference",
        "alpaca_gpt4_preference",
        "alpaca_noisy_multi_preference",
    ] = field(
        default="alpaca_noisy_multi_preference",
        metadata={
            "help": "Name of the dataset. Fetches the human or GPT-4 preference data."
        },
    )

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="dpo-out",
            evaluation_strategy="steps",
            bf16=True,
            remove_unused_columns=False,
            run_name="dpo_llama2",
        )
    )


def get_alpaca_farm_reward_dataset(
    args,
    sanity_check: bool = False,
    num_proc=24,
) -> Dataset:
    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split="preference",
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples):
        instructions = samples["instruction"]
        _inputs = samples["input"]
        output_1s = samples["output_1"]
        output_2s = samples["output_2"]
        preferences = samples["preference"]
        prompts = []
        chosens = []
        rejecteds = []
        for i, instruction in enumerate(instructions):
            _input = _inputs[i]
            output_1 = output_1s[i]
            output_2 = output_2s[i]
            preference = preferences[i]
            if preference == 1:
                chosen = output_1
                rejected = output_2
            else:
                chosen = output_2
                rejected = output_1
            prompt = f"Instruction:\n{instruction}\n\nInput:\n{_input}\n\nAnswer:\n"
            prompts.append(prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)

        return {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    script_args = tyro.cli(ScriptArguments)
    training_args = script_args.training_args
    assert training_args.bf16

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token

    # 2. Load the dataset
    dataset = get_alpaca_farm_reward_dataset(
        script_args, sanity_check=script_args.sanity_check
    )
    dataset = dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)
