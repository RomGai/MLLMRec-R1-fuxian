import argparse
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from utils.data_loader import RecSFTDataset



def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training for Rec System")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/MLLMRec-R1", help="Dataset root path")
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model id or local path. Defaults to direct HF loading.",
    )
    parser.add_argument("--use_cot", action="store_true", help="Enable Chain-of-Thought (CoT) reasoning in SFT prompts")
    parser.add_argument("--cot_prob", type=float, default=0.1, help="cot_prob")
    parser.add_argument("--min_inter", type=int, default=10, help="user sequence length")
    parser.add_argument("--num_neg", type=int, default=9, help="negative samples per user")
    parser.add_argument("--tag", type=str, required=True, help="LoRA adapter output sub-dir name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=1)

    return parser.parse_args()


class DetailedProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.max_steps:
            pct = 100.0 * state.global_step / state.max_steps
            print(f"[SFT][Step {state.global_step}/{state.max_steps}] progress={pct:.2f}%")


def main():
    args = parse_args()

    print(f"[SFT] Step 1/6: Loading tokenizer from: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    print("[SFT] Step 2/6: Loading backbone model")
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.train()

    print("[SFT] Step 3/6: Configuring LoRA")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("[SFT] Step 4/6: Building training dataset")
    train_ds = RecSFTDataset(
        root_path=args.root,
        dataset_name=args.dataset,
        split="train",
        num_neg=args.num_neg,
        min_interactions=args.min_inter,
        mode="train",
        seed=42,
        use_cot=args.use_cot,
        cot_prob=args.cot_prob,
    )
    print(f"[SFT] Dataset size: {len(train_ds)}")
    print("[SFT] Example sample:")
    print(train_ds[0]["messages"])

    hf_train_ds = Dataset.from_list([train_ds[i] for i in range(len(train_ds))])

    output_dir = f"{args.root}/checkpoints/{args.dataset}/{args.tag}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=100,
        save_total_limit=20,
        bf16=True,
        completion_only_loss=False,
        dataset_text_field="messages",
    )

    print("[SFT] Step 5/6: Initializing trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_train_ds,
        args=sft_config,
        peft_config=lora_config,
        callbacks=[DetailedProgressCallback()],
    )

    print("[SFT] Step 6/6: Start training")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SFT][DONE] Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
