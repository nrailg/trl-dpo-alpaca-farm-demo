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


class MyDPODataCollatorWithPadding(DPODataCollatorWithPadding):
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        batch = {}
        assert not self.is_encoder_decoder

        assert isinstance(prompt, str)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        ptids = prompt_tokens["input_ids"]
        ptam = prompt_tokens["attention_mask"]
        assert len(ptids) == len(ptam)

        assert isinstance(chosen, str)
        chosen_tokens = self.tokenizer(prompt + chosen, add_special_tokens=False)
        assert ptids == chosen_tokens["input_ids"][: len(ptids)]
        chosen_tokens["input_ids"] = chosen_tokens["input_ids"][len(ptids) :]
        chosen_tokens["attention_mask"] = chosen_tokens["attention_mask"][len(ptids) :]

        assert isinstance(rejected, str)
        rejected_tokens = self.tokenizer(prompt + rejected, add_special_tokens=False)
        assert ptids == rejected_tokens["input_ids"][: len(ptids)]
        rejected_tokens["input_ids"] = rejected_tokens["input_ids"][len(ptids) :]
        rejected_tokens["attention_mask"] = rejected_tokens["attention_mask"][
            len(ptids) :
        ]

        # add EOS token to end of prompt
        prompt_tokens["input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens[
            "input_ids"
        ]
        prompt_tokens["attention_mask"] = [1] + prompt_tokens["attention_mask"]
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)
        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        # if combined sequence is too long, truncate the prompt
        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            prompt_tokens = {
                k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
            }
        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in chosen_tokens.items()
            }
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in rejected_tokens.items()
            }

        # Create labels
        chosen_sequence_tokens = {
            k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
        }
        rejected_sequence_tokens = {
            k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected
        return batch


def anal_dataset(dataset, data_collator):
    dataloader = DataLoader(train_dataset, collate_fn=data_collator)
    pls = []
    cls = []
    rls = []
    for xi, x in enumerate(dataloader):
        """
        if torch.distributed.get_rank() == 0:
            pprint.pprint(x)
        if xi == 2:
            break
        """
        if torch.distributed.get_rank() == 0:
            pls.append(x["prompt_input_ids"].shape[1])
            cls.append(x["chosen_input_ids"].shape[1])
            rls.append(x["rejected_input_ids"].shape[1])

    pls = sorted(pls)
    cls = sorted(cls)
    rls = sorted(rls)
    if torch.distributed.get_rank() == 0:
        '''
        print("pls", pls)
        print("cls", cls)
        print("rls", rls)
        '''
        print("pls max", pls[-1])
        print("cls max", cls[-1])
        print("rls max", rls[-1])


class MyTrainer(DPOTrainer):
    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        assert not self.use_dpo_data_collator
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return (loss, metrics)
        return loss


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
    my_data_collator = MyDPODataCollatorWithPadding(
        tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    #anal_dataset(dataset, my_data_collator)

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
    dpo_trainer = MyTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        data_collator=my_data_collator,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # 7. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
