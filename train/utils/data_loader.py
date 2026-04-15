# utils/data_loader.py
import json
import os
import random
import re
import zipfile

import pandas as pd
from torch.utils.data import Dataset


def _read_table_auto(path: str) -> pd.DataFrame:
    """Read csv/tsv file with robust separator inference."""
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


class InteractionDataset(Dataset):
    """
    Basic interaction dataset.

    Supports two layouts:
    1) Original layout: data/{dataset}/train.tsv, test.tsv
    2) Flat Amazon layout:
       data/{dataset}_user_items_negs_train.csv
       data/{dataset}_user_items_negs_test.csv
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "movielens",
        split: str = "train",
        min_interactions: int = 10,
        history_len: int = 9,
    ):
        self.samples = []
        self._build_samples(
            root_path=root_path,
            dataset_name=dataset_name,
            split=split,
            min_interactions=min_interactions,
            history_len=history_len,
        )

    def _build_samples(self, root_path, dataset_name, split, min_interactions, history_len):
        original_tsv_path = f"{root_path}/data/{dataset_name}/{split}.tsv"
        amazon_csv_path = f"{root_path}/data/{dataset_name}_user_items_negs_{split}.csv"

        if os.path.exists(original_tsv_path):
            df = pd.read_csv(
                original_tsv_path,
                sep="\t",
                header=None,
                names=["user_id", "item_seq"],
            )
            for _, row in df.iterrows():
                items = str(row["item_seq"]).split()
                if len(items) < min_interactions:
                    continue
                seq = items[-(history_len + 1) :]
                self.samples.append(
                    {
                        "user_id": str(row["user_id"]),
                        "history_ids": [str(x) for x in seq[:-1]],
                        "target_id": str(seq[-1]),
                    }
                )
            print(
                f"[InteractionDataset] Loaded original split={split}, "
                f"users={len(self.samples)} from {original_tsv_path}"
            )
            return

        if not os.path.exists(amazon_csv_path):
            raise FileNotFoundError(
                f"Cannot find split file for dataset='{dataset_name}', split='{split}'. "
                f"Tried:\n- {original_tsv_path}\n- {amazon_csv_path}"
            )

        # File format (no header):
        # user_id \t seq_items_csv \t neg_items_csv
        df = pd.read_csv(
            amazon_csv_path,
            sep="\t",
            header=None,
            names=["user_id", "item_seq_csv", "neg_items_csv"],
        )

        for _, row in df.iterrows():
            seq_items = [x for x in str(row["item_seq_csv"]).split(",") if x != ""]
            if len(seq_items) < 2:
                continue
            if len(seq_items) < min_interactions:
                continue

            seq = seq_items[-(history_len + 1) :]
            self.samples.append(
                {
                    "user_id": str(row["user_id"]),
                    "history_ids": [str(x) for x in seq[:-1]],
                    "target_id": str(seq[-1]),
                }
            )

        print(
            f"[InteractionDataset] Loaded amazon split={split}, "
            f"users={len(self.samples)} from {amazon_csv_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RecSFTDataset(Dataset):
    """SFT dataset for recommendation with fixed negative sampling and caching."""

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "movielens",
        split: str = "train",
        num_neg: int = 9,
        min_interactions: int = 10,
        mode: str = "train",
        seed: int = 42,
        use_cot: bool = False,
        cot_prob: float = 0.1,
    ):
        self.root_path = root_path
        self.cot_path = f"{root_path}/data/{dataset_name}/{dataset_name}_deepseek_cot.json"
        self.dataset_name = dataset_name
        self.split = split
        self.num_neg = num_neg
        self.min_interactions = min_interactions
        self.history_len = min_interactions - 1
        self.cot_prob = cot_prob
        self.mode = mode
        self.seed = seed

        self.processed_dir = f"{root_path}/data/{dataset_name}/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

        if use_cot:
            self.cache_file = os.path.join(
                self.processed_dir,
                f"{split}_{mode}_seed{seed}_neg{num_neg}_cot_{cot_prob}_int_{min_interactions}.json",
            )
        else:
            self.cache_file = os.path.join(
                self.processed_dir,
                f"{split}_{mode}_seed{seed}_neg{num_neg}_int_{min_interactions}.json",
            )

        if os.path.exists(self.cache_file):
            print(f"[LOAD FIXED DATA] {self.cache_file}")
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
            return

        print(
            f"[BUILD DATA FIRST TIME] split={split}, mode={mode}, seed={seed}, "
            f"num_neg={num_neg}, cot={cot_prob}, int={min_interactions}"
        )
        self.rng = random.Random(seed)

        self.inter_ds = InteractionDataset(
            root_path=root_path,
            dataset_name=dataset_name,
            split=split,
            min_interactions=min_interactions,
            history_len=self.history_len,
        )

        self.id2title = self._load_item_text(root_path, dataset_name)

        self.idx2reasoning = {}
        if os.path.exists(self.cot_path):
            with open(self.cot_path, "r", encoding="utf-8") as f:
                cot_data = json.load(f)
            for row in cot_data:
                if "line_idx" in row:
                    self.idx2reasoning[int(row["line_idx"])] = str(row.get("reasoning", ""))

        self.all_item_ids = list(self.id2title.keys())
        self.samples = []
        self._build_all_samples()

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, ensure_ascii=False, indent=2)
        print(f"[SAVE FIXED DATA] {self.cache_file}")

    def _load_item_text(self, root_path: str, dataset_name: str):
        titles_path = f"{root_path}/data/{dataset_name}/{dataset_name}_titles.csv"
        if os.path.exists(titles_path):
            titles_df = _read_table_auto(titles_path)
            item_col = titles_df.columns[0]
            title_col = titles_df.columns[1]
            return {str(row[item_col]): str(row[title_col]) for _, row in titles_df.iterrows()}

        desc_tsv = f"{root_path}/data/{dataset_name}_item_desc.tsv"
        desc_zip = f"{root_path}/data/{dataset_name}_item_desc.zip"
        if os.path.exists(desc_tsv):
            desc_df = _read_table_auto(desc_tsv)
        elif os.path.exists(desc_zip):
            with zipfile.ZipFile(desc_zip) as zf:
                tsv_name = next((x for x in zf.namelist() if x.endswith(".tsv")), None)
                if tsv_name is None:
                    raise FileNotFoundError(f"No .tsv found in {desc_zip}")
                with zf.open(tsv_name) as f:
                    desc_df = pd.read_csv(f, sep="\t")
        else:
            raise FileNotFoundError(
                f"Cannot find item text file for dataset '{dataset_name}'. "
                f"Tried {titles_path}, {desc_tsv}, {desc_zip}"
            )

        text_col = "summary" if "summary" in desc_df.columns else desc_df.columns[1]
        id2title = {}
        for _, row in desc_df.iterrows():
            iid = str(row["item_id"])
            txt = str(row[text_col]).strip()
            if not txt or txt.lower() == "nan":
                txt = f"Item {iid}"
            id2title[iid] = txt
        return id2title

    def _sample_negatives(self, history_ids, target_id):
        forbidden = set(history_ids + [target_id])
        neg_ids = []
        while len(neg_ids) < self.num_neg:
            cand = self.rng.choice(self.all_item_ids)
            if cand not in forbidden:
                neg_ids.append(cand)
                forbidden.add(cand)
        return neg_ids

    def extract_step(self, reasoning: str) -> str:
        if not reasoning:
            return ""
        pattern = r"Step\s*3\s*:(.*?)(?:Step\s*4\s*:|$)"
        match = re.search(pattern, reasoning, flags=re.S | re.I)
        if not match:
            return ""
        step = match.group(1).strip()
        step = re.sub(r"^Retrieval criteria\s*:\s*", "", step, flags=re.I)
        step = re.sub(r"\s+", " ", step).strip()
        return step

    def _build_prompt(self, history_ids, candidates, reasoning):
        lines = [
            "You are a movie recommendation assistant.",
            "Here are the user's most recently watched movies sequence:",
        ]
        for i, iid in enumerate(history_ids, 1):
            title = self.id2title.get(iid, "Unknown Title")
            lines.append(f"{i}. [ITEM_{iid}] {title}")

        lines.extend(["", "Candidate movies:"])
        for iid in candidates:
            title = self.id2title.get(iid, "Unknown Title")
            lines.append(f"[ITEM_{iid}] {title}")

        lines.append("")
        if reasoning and random.random() < self.cot_prob:
            lines.append("Possible reasoning process to help predict the next item, for reference ONLY.")
            lines.append("<think>")
            lines.append(reasoning)
            lines.append("</think>")

        lines.extend(
            [
                "",
                "Now answer the following question.\n"
                "Only output in format [ITEM_xxxx] Movie Title.\n"
                "No additional caption, summary, comments, or explanation.",
            ]
        )
        return "\n".join(lines)

    def _build_all_samples(self):
        total = len(self.inter_ds)
        print(f"[RecSFTDataset] Building {total} samples for split={self.split}, mode={self.mode}")

        for sample_idx, rec in enumerate(self.inter_ds):
            history_ids = rec["history_ids"]
            target_id = rec["target_id"]

            neg_ids = self._sample_negatives(history_ids, target_id)
            candidates = [str(target_id)] + [str(nid) for nid in neg_ids]
            self.rng.shuffle(candidates)

            reasoning = self.idx2reasoning.get(sample_idx, "")
            prompt = self._build_prompt(history_ids, candidates, reasoning)

            target_title = self.id2title.get(target_id, "Unknown Title")
            completion = f"[ITEM_{target_id}] {target_title}"

            if self.mode == "train":
                self.samples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ]
                    }
                )
            elif self.mode == "generate":
                self.samples.append(
                    {
                        "prompt": prompt,
                        "target_id": str(target_id),
                        "candidates": [str(cid) for cid in candidates],
                    }
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            if (sample_idx + 1) % 200 == 0 or (sample_idx + 1) == total:
                print(
                    f"[RecSFTDataset] processed {sample_idx + 1}/{total} "
                    f"({(sample_idx + 1) / max(total, 1) * 100:.2f}%)"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RecGRPODataset(Dataset):
    """Dataset wrapper for GRPO training in recommendation setting."""

    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        split: str = "train",
        num_neg: int = 9,
        min_interactions: int = 10,
        seed: int = 42,
        rec_top_k: int = 1,
        add_format_instruction: bool = True,
        instruction_template: str | None = None,
        use_cot: bool = False,
        cot_prob: float = 0.1,
    ) -> None:
        super().__init__()

        base_ds = RecSFTDataset(
            root_path=root_path,
            dataset_name=dataset_name,
            split=split,
            num_neg=num_neg,
            min_interactions=min_interactions,
            mode="generate",
            seed=seed,
            use_cot=use_cot,
            cot_prob=cot_prob,
        )

        self.samples = []
        self.rec_top_k = rec_top_k
        self.add_format_instruction = add_format_instruction

        if instruction_template is None:
            instruction_template = (
                "\n\n[OUTPUT GUIDELINES]\n"
                "You MUST answer with only one movie from candidate movies.\n"
                "The output token MUST be in the form:\n"
                "[ITEM_xxxx] Movie Title\n\n"
                "If you want to provide reasoning, you may append it ONLY after answer\n"
                "in a single line using:\n"
                "<think> your reasoning here </think>\n\n"
            )

        self.instruction_template = instruction_template

        for ex in base_ds:
            prompt = ex["prompt"]
            if add_format_instruction:
                prompt += instruction_template

            self.samples.append(
                {
                    "prompt": prompt,
                    "target_id": ex["target_id"],
                    "candidates": ex["candidates"],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
