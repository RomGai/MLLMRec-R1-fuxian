import argparse
import json
import math
import os
import random
import re

import pandas as pd
import torch
import torch.distributed as dist
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loader import RecSFTDataset



METRIC_CUTOFFS = [10, 20, 40]


def parse_args():
    parser = argparse.ArgumentParser(description="Ranking eval with 1 target + random negatives")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/MLLMRec-R1", help="Project root path")
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF backbone id when --sft_tag is not provided",
    )
    parser.add_argument("--min_inter", type=int, default=10, help="user sequence length")
    parser.add_argument("--num_rand", type=int, default=1000, help="number of random negatives per user")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu (non-distributed only)")
    parser.add_argument("--sft_tag", type=str, default=None, help="Optional local SFT checkpoint tag")
    parser.add_argument("--tag", type=str, default=None, help="Optional LoRA adapter dir under checkpoints/<dataset>/")
    parser.add_argument("--distributed", action="store_true", help="Use torchrun multi-GPU data parallel inference")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_user_messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def _init_distributed_if_needed(args):
    distributed = bool(args.distributed) and ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return distributed, rank, world_size, local_rank, device

    return False, 0, 1, 0, args.device


def _sample_eval_candidates(ex, all_item_ids, num_rand, rng):
    target_id = str(ex["target_id"])

    history_ids = set(re.findall(r"\[ITEM_(\d+)\]", ex["prompt"]))
    forbidden = set(history_ids)
    forbidden.add(target_id)

    needed = min(num_rand, max(0, len(all_item_ids) - len(forbidden)))
    sampled = []
    sampled_set = set()
    while len(sampled) < needed:
        cand = rng.choice(all_item_ids)
        if cand in forbidden or cand in sampled_set:
            continue
        sampled.append(cand)
        sampled_set.add(cand)

    candidates = [target_id] + sampled
    rng.shuffle(candidates)
    return candidates


def _build_ranking_prompt(prompt_base: str, candidates):
    lines = [prompt_base, "", "[RANKING TASK]", "Please rank ALL candidate items from best to worst."]
    lines.append("Output item ids only, one per line, format: [ITEM_xxxx]")
    lines.append("Rank list:")
    for cid in candidates:
        lines.append(f"[ITEM_{cid}]")
    return "\n".join(lines)


def generate_ranked_items(model, tokenizer, prompt: str, device: str, max_count: int):
    user_messages = build_user_messages(prompt)
    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=min(4096, 8 * max_count),
        do_sample=False,
        top_p=0.7,
        temperature=0.1,
    )

    gen = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    ids = re.findall(r"ITEM[_\s-]?(\d+)", gen)

    ranked = []
    seen = set()
    for item in ids:
        if item not in seen:
            ranked.append(item)
            seen.add(item)
        if len(ranked) >= max_count:
            break
    return ranked


def _metric_update(ranked_ids, target_id, agg):
    try:
        rank_pos = ranked_ids.index(target_id) + 1
    except ValueError:
        rank_pos = None

    for k in METRIC_CUTOFFS:
        hit = 1 if rank_pos is not None and rank_pos <= k else 0
        agg[f"hit@{k}"] += hit
        if hit:
            agg[f"ndcg@{k}"] += 1.0 / math.log2(rank_pos + 1)

    agg["count"] += 1
    return rank_pos


def _metric_avg_str(agg):
    c = max(1, agg["count"])
    parts = []
    for k in METRIC_CUTOFFS:
        hr = agg[f"hit@{k}"] / c
        ndcg = agg[f"ndcg@{k}"] / c
        parts.append(f"HR@{k}={hr:.4f}, NDCG@{k}={ndcg:.4f}")
    return " | ".join(parts)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    distributed, rank, world_size, _, device = _init_distributed_if_needed(args)

    if args.sft_tag:
        base_model = f"{args.root}/checkpoints/{args.dataset}/{args.sft_tag}"
        print(f"[INF] Step 1/6: Using local SFT model {base_model}")
    else:
        base_model = args.backbone
        print(f"[INF] Step 1/6: Using HF backbone {base_model}")

    print("[INF] Step 2/6: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INF] Step 3/6: Loading model")
    if distributed:
        model_base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    else:
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = model_base
    if args.tag:
        adapter_dir = f"{args.root}/checkpoints/{args.dataset}/{args.tag}"
        print(f"[INF] Step 4/6: Loading LoRA adapter {adapter_dir}")
        model = PeftModel.from_pretrained(model_base, adapter_dir)
    else:
        print("[INF] Step 4/6: No LoRA adapter, evaluate base model directly")

    model.eval()
    if not distributed:
        model.to(device)

    print("[INF] Step 5/6: Building test dataset")
    test_ds = RecSFTDataset(
        root_path=args.root,
        dataset_name=args.dataset,
        split="test",
        num_neg=1,
        min_interactions=args.min_inter,
        mode="generate",
        seed=args.seed,
    )
    all_item_ids = list(test_ds.id2title.keys())
    print(f"[INF] Test users={len(test_ds)}, item_pool={len(all_item_ids)}")

    indices = list(range(rank, len(test_ds), world_size))
    iterator = tqdm(indices, desc=f"Ranking eval (world_size={world_size})") if ((not distributed) or rank == 0) else indices

    agg_local = {"count": 0, "hit@10": 0, "hit@20": 0, "hit@40": 0, "ndcg@10": 0.0, "ndcg@20": 0.0, "ndcg@40": 0.0}
    results_local = []

    print("[INF] Step 6/6: Start per-user ranking inference")
    for local_i, idx in enumerate(iterator, 1):
        ex = test_ds[idx]
        target_id = str(ex["target_id"])

        candidates = _sample_eval_candidates(ex, all_item_ids, args.num_rand, rng)
        ranking_prompt = _build_ranking_prompt(ex["prompt"], candidates)
        ranked_ids = generate_ranked_items(
            model=model,
            tokenizer=tokenizer,
            prompt=ranking_prompt,
            device=device,
            max_count=len(candidates),
        )

        rank_pos = _metric_update(ranked_ids, target_id, agg_local)
        running = _metric_avg_str(agg_local)
        print(
            f"[INF][rank={rank}] user_idx={idx} processed={agg_local['count']} "
            f"target_rank={rank_pos} | running_avg: {running}"
        )

        results_local.append(
            {
                "index": idx,
                "target_id": target_id,
                "num_candidates": len(candidates),
                "target_rank": rank_pos,
                "pred_ranked": ranked_ids,
            }
        )

    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, {"results": results_local, "agg": agg_local})

        if rank == 0:
            all_results = []
            total_agg = {"count": 0, "hit@10": 0, "hit@20": 0, "hit@40": 0, "ndcg@10": 0.0, "ndcg@20": 0.0, "ndcg@40": 0.0}
            for part in gathered:
                all_results.extend(part["results"])
                for k, v in part["agg"].items():
                    total_agg[k] += v
        else:
            all_results = None
            total_agg = None
    else:
        all_results = results_local
        total_agg = agg_local

    if (not distributed) or rank == 0:
        total = max(1, total_agg["count"])
        summary = {
            "dataset": args.dataset,
            "num_rand": args.num_rand,
            "total_test": total_agg["count"],
            "HR@10": total_agg["hit@10"] / total,
            "HR@20": total_agg["hit@20"] / total,
            "HR@40": total_agg["hit@40"] / total,
            "NDCG@10": total_agg["ndcg@10"] / total,
            "NDCG@20": total_agg["ndcg@20"] / total,
            "NDCG@40": total_agg["ndcg@40"] / total,
            "distributed": bool(distributed),
            "world_size": world_size,
        }

        print("\n[INF][FINAL] " + " | ".join([f"{k}={v:.4f}" for k, v in summary.items() if isinstance(v, float)]))

        result_dir = f"{args.root}/result/{args.dataset}"
        os.makedirs(result_dir, exist_ok=True)

        tag_name = args.tag if args.tag else "base"
        save_prefix = f"{tag_name}_{args.dataset}_rand{args.num_rand}"

        save_csv = os.path.join(result_dir, f"{save_prefix}.csv")
        pd.DataFrame(all_results).to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"[INF][Saved Results] {save_csv}")

        save_json = os.path.join(result_dir, f"{save_prefix}.json")
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INF][Saved Metrics] {save_json}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
