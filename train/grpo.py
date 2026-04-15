# grpo_train.py
import argparse
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from utils.data_loader import RecGRPODataset


def parse_args():
    p = argparse.ArgumentParser("GRPO training for Recommendation System")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--root", type=str, default="/root/autodl-tmp/MLLMRec-R1")
    p.add_argument("--min_inter", type=int, default=10)
    p.add_argument("--num_neg", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sft_tag", type=str, default=None, help="Optional local SFT checkpoint tag")
    p.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF backbone id when --sft_tag is not provided",
    )
    p.add_argument("--use_cot", action="store_true", help="Enable Chain-of-Thought (CoT) reasoning in SFT prompts")
    p.add_argument("--cot_prob", type=float, default=0.1, help="cot_prob")

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=100)

    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.1)

    p.add_argument("--tag", type=str, default="qwen3_4b_grpo_lora")
    return p.parse_args()


class DetailedProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.max_steps:
            pct = 100.0 * state.global_step / state.max_steps
            print(f"[GRPO][Step {state.global_step}/{state.max_steps}] progress={pct:.2f}%")


ITEM_PATTERN = re.compile(r"^\[ITEM_(\d+)\]\s+(.+)$")


def rec_reward_func(completions, target_id, candidates, **kwargs):
    rewards = []
    for comp, tid in zip(completions, target_id):
        text = comp.strip()
        if "</think>" in text:
            text = text.split("</think>")[0].strip()

        m = ITEM_PATTERN.match(text)
        if not m:
            rewards.append(-1.0)
            continue

        pred_id = m.group(1)
        rewards.append(1.0 if pred_id == tid else 0.3)
    return rewards


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bsz = world_size * args.batch * args.accum
    if effective_bsz % args.num_generations != 0:
        raise ValueError(
            f"Effective batch size (WORLD_SIZE*batch*accum)={effective_bsz} "
            f"must be divisible by num_generations={args.num_generations}."
        )

    if args.sft_tag:
        base_model_path = f"{args.root}/checkpoints/{args.dataset}/{args.sft_tag}"
        print(f"[GRPO] Step 1/6: Using local SFT model at {base_model_path}")
    else:
        base_model_path = args.backbone
        print(f"[GRPO] Step 1/6: Using HF backbone {base_model_path}")

    print("[GRPO] Step 2/6: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[GRPO] Step 3/6: Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.train()

    print("[GRPO] Step 4/6: Building GRPO dataset")
    train_ds = RecGRPODataset(
        root_path=args.root,
        dataset_name=args.dataset,
        split="train",
        num_neg=args.num_neg,
        min_interactions=args.min_inter,
        seed=args.seed,
        rec_top_k=1,
        add_format_instruction=True,
        use_cot=args.use_cot,
        cot_prob=args.cot_prob,
    )
    hf_train_ds = Dataset.from_list([train_ds[i] for i in range(len(train_ds))])
    print(f"[GRPO] Dataset size: {len(hf_train_ds)}")
    print(hf_train_ds[0])

    output_dir = f"{args.root}/checkpoints/{args.dataset}/{args.tag}"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
    )

    print("[GRPO] Step 5/6: Initializing trainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_train_ds,
        reward_funcs=rec_reward_func,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[DetailedProgressCallback()],
    )

    print("[GRPO] Step 6/6: Start training")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[GRPO][DONE] Training finished. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
