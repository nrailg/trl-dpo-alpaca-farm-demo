# trl-dpo-alpaca-farm-demo

This project trains llama2 with DPO on [Alpaca-farm dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/viewer/alpaca_noisy_multi_preference).

This project is just a demo for testing purpose. I prefer deepspeed to peft, so I decided to
create a new program for it.

## SFT

```bash
accelerate launch --config_file ${YAML_TO} sft_llama2.py \
    --model_name /home/nrwu/model-ckpt/Llama-2-7b-hf \
    --dataset_path /home/nrwu/work/hf-hub/tatsu-lab/alpaca_farm \
    --training_args.output_dir sft-out \
    --streaming False \
    --training_args.gradient_checkpointing \
    --training_args.save_steps 155 \
    --training_args.logging_steps 1 \
    --training_args.remove_unused_columns False \
    --training_args.report_to None \
    --training_args.bf16 \
    --training_args.num_train_epochs 3 \
    --training_args.max_steps -1 \
    --training_args.per_device_train_batch_size 2 \
    --training_args.per_device_eval_batch_size 2 \
    --training_args.gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length 1024 \
    --training_args.weight_decay 0. \
    --training_args.warmup_ratio 0.03 \
    --training_args.learning_rate 2e-5
```

## DPO

```bash
accelerate launch --config_file ${YAML_TO} dpo_llama2.py \
    --model_name_or_path ./sft-out \
    --dataset_path /home/nrwu/work/hf-hub/tatsu-lab/alpaca_farm \
    --beta 0.2 \
    --training_args.output_dir dpo-beta-0.2-out \
    --max_prompt_length 512 \
    --max_length 1024 \
    --training_args.gradient_checkpointing \
    --training_args.save_steps 150 \
    --training_args.logging_steps 1 \
    --training_args.remove_unused_columns False \
    --training_args.report_to None \
    --training_args.bf16 \
    --training_args.num_train_epochs 3 \
    --training_args.per_device_train_batch_size 1 \
    --training_args.per_device_eval_batch_size 1 \
    --training_args.gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --training_args.weight_decay 0. \
    --training_args.warmup_ratio 0.03 \
    --training_args.learning_rate 2e-5
```

## Evalution

I evaluate DPO model with GPT-4 against SFT model.
It's tested with a closed source tool in Tencent, so I can't give you the procedure. But it's
very similar to Vicuna chatbot arena, I guess you can make similar evaluations yourself.

I tested 80 queries.
| All             | 80   |
| --------------- | ---- |
| DPO vs SFT Win  | 56   |
| DPO vs SFT Draw | 4    |
| DPO vs SFT Lose | 20   |
