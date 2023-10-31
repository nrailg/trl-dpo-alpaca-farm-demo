import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})

    dataset_path: Optional[str] = field(default="tatsu-lab/alpaca_farm",
                                        metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    dataset_seed: Optional[int] = field(default=42, metadata={'help': 'dataset shuffle seed'})

    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use packing for SFTTrainer"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir='output',
        ))


def formatting_func(example):
    instructions = example['instruction']
    _inputs = example['input']
    outputs = example['output']
    texts = []
    for i, instruction in enumerate(instructions):
        _input = _inputs[i]
        output = outputs[i]
        text = f"Instruction:\n{instruction}\n\nInput:\n{_input}\n\nAnswer:\n{output}"
        texts.append(text)
    return texts


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_path,
        name='alpaca_instructions',
        split='sft',
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=script_args.dataset_seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=script_args.dataset_seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    return train_data, valid_data


if __name__ == '__main__':
    script_args = tyro.cli(ScriptArguments)
    training_args = script_args.training_args
    assert training_args.bf16
    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

    trainer = SFTTrainer(
        args=training_args,
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=script_args.packing,
        formatting_func=formatting_func,
        max_seq_length=script_args.max_seq_length,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
