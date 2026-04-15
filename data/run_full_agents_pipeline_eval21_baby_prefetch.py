from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from run_full_agents_pipeline_eval21_baby import build_argparser as build_base_argparser
from run_full_agents_pipeline_eval21_baby import main as run_base_main


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _is_probable_url(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_suffix_from_url(url: str, content_type: Optional[str] = None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix

    if content_type is not None:
        normalized_content_type = content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_extension(normalized_content_type)
        if guessed in IMAGE_EXTENSIONS:
            return guessed

    return ".jpg"


def _image_cache_path(cache_dir: Path, image_url: str, item_id: str) -> Path:
    digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
    suffix = _safe_suffix_from_url(image_url)
    safe_item = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(item_id))
    return cache_dir / f"{safe_item}_{digest}{suffix}"


def _read_item_rows(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            rows.append(
                {
                    "item_id": item_id,
                    "image": str(row.get("image", "")).strip(),
                    "summary": str(row.get("summary", "")),
                }
            )
    return rows


def _write_item_rows(path: str | Path, rows: Sequence[Dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "image", "summary"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _collect_prefetch_jobs(rows: Sequence[Dict[str, str]], cache_dir: Path) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        image_value = str(row.get("image", "")).strip()
        if not image_value or not _is_probable_url(image_value):
            continue
        if image_value in seen:
            continue
        seen.add(image_value)
        jobs.append((row["item_id"], image_value, _image_cache_path(cache_dir, image_value, row["item_id"])))
    return jobs


def _download_one(item_id: str, image_url: str, local_path: Path, timeout: int) -> Tuple[str, str, str, Optional[str]]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return item_id, image_url, str(local_path), None

    try:
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0 Codex Image Prefetch"})
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"HTTP {status}")
            suffix = _safe_suffix_from_url(image_url, response.headers.get("Content-Type"))
            final_path = local_path if local_path.suffix.lower() == suffix else local_path.with_suffix(suffix)
            if final_path.exists() and final_path.stat().st_size > 0:
                return item_id, image_url, str(final_path), None
            with final_path.open("wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 256)
        return item_id, image_url, str(final_path), None
    except Exception as exc:
        return item_id, image_url, "", f"{type(exc).__name__}: {exc}"


def _prefetch_images(jobs: Sequence[Tuple[str, str, Path]], max_workers: int, timeout: int) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not jobs:
        return mapping

    total = len(jobs)
    print(f"[Prefetch] Downloading/checking {total} unique images into local cache")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_one, item_id, image_url, local_path, timeout): (item_id, image_url)
            for item_id, image_url, local_path in jobs
        }
        for future in as_completed(future_map):
            item_id, image_url = future_map[future]
            completed += 1
            _item_id, url, local_path, error = future.result()
            if error:
                print(f"[Prefetch][{completed}/{total}] FAIL item_id={item_id} url={image_url} error={error}")
                continue
            mapping[url] = local_path
            print(f"[Prefetch][{completed}/{total}] OK item_id={item_id} local_path={local_path}")
    return mapping


def _rewrite_rows_with_local_images(rows: Sequence[Dict[str, str]], url_to_local: Dict[str, str]) -> List[Dict[str, str]]:
    rewritten: List[Dict[str, str]] = []
    for row in rows:
        image_value = str(row.get("image", "")).strip()
        replacement = image_value
        if image_value:
            if _is_probable_url(image_value):
                replacement = url_to_local.get(image_value, image_value)
            else:
                image_path = Path(image_value)
                if image_path.exists():
                    replacement = str(image_path)
        rewritten.append(
            {
                "item_id": row["item_id"],
                "image": replacement,
                "summary": row.get("summary", ""),
            }
        )
    return rewritten


def build_argparser() -> argparse.ArgumentParser:
    parser = build_base_argparser()
    parser.description = "Eval21 baby runner with upfront local image prefetch for Agent1/2"
    parser.add_argument(
        "--image-cache-dir",
        default="",
        help="Directory for downloaded images. Defaults to <eval-run-root>/prefetched_images.",
    )
    parser.add_argument(
        "--prefetch-work-dir",
        default="",
        help="Directory for rewritten local item_desc TSV files. Defaults to <eval-run-root>/prefetch_inputs.",
    )
    parser.add_argument("--prefetch-max-workers", type=int, default=16)
    parser.add_argument("--prefetch-timeout", type=int, default=60)
    return parser


def main(args: argparse.Namespace) -> None:
    eval_run_root = Path(args.eval_run_root)
    cache_dir = Path(args.image_cache_dir) if str(args.image_cache_dir).strip() else eval_run_root / "prefetched_images"
    work_dir = Path(args.prefetch_work_dir) if str(args.prefetch_work_dir).strip() else eval_run_root / "prefetch_inputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    item_desc_rows = _read_item_rows(args.item_desc_tsv)
    agent2_item_desc_path = str(args.agent2_item_desc_tsv or args.item_desc_tsv)
    agent2_rows: List[Dict[str, str]] = []
    if Path(agent2_item_desc_path).resolve() == Path(args.item_desc_tsv).resolve():
        agent2_rows = item_desc_rows
    else:
        agent2_rows = _read_item_rows(agent2_item_desc_path)

    jobs = _collect_prefetch_jobs(item_desc_rows, cache_dir)
    if agent2_rows is not item_desc_rows:
        existing_urls = {url for _, url, _ in jobs}
        for job in _collect_prefetch_jobs(agent2_rows, cache_dir):
            if job[1] not in existing_urls:
                jobs.append(job)
                existing_urls.add(job[1])

    url_to_local = _prefetch_images(
        jobs=jobs,
        max_workers=max(1, int(args.prefetch_max_workers)),
        timeout=max(1, int(args.prefetch_timeout)),
    )

    local_item_desc = work_dir / f"{Path(args.item_desc_tsv).stem}.local.tsv"
    _write_item_rows(local_item_desc, _rewrite_rows_with_local_images(item_desc_rows, url_to_local))

    if agent2_rows is item_desc_rows:
        local_agent2_item_desc = local_item_desc
    else:
        local_agent2_item_desc = work_dir / f"{Path(agent2_item_desc_path).stem}.local.tsv"
        _write_item_rows(local_agent2_item_desc, _rewrite_rows_with_local_images(agent2_rows, url_to_local))

    print(
        f"[Prefetch] Prepared local item_desc TSVs:\n"
        f"  item_desc_tsv={local_item_desc}\n"
        f"  agent2_item_desc_tsv={local_agent2_item_desc}\n"
        f"  cache_dir={cache_dir}"
    )

    forwarded = argparse.Namespace(**vars(args))
    forwarded.item_desc_tsv = str(local_item_desc)
    forwarded.agent2_item_desc_tsv = str(local_agent2_item_desc)
    run_base_main(forwarded)


if __name__ == "__main__":
    main(build_argparser().parse_args())
