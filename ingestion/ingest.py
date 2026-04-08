import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_utils import BASE_MODEL_NAME, CATEGORIES, CHECKPOINT_DIR, LOCAL_BACKBONE, MODEL_REPO, INDEX_REPO, PaperClassifier

TARGET_CATEGORIES = [category for category in CATEGORIES if category != "other"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def disable_socks_proxy():
    removed = []
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.lower().startswith("socks"):
            removed.append((key, value))
            os.environ.pop(key, None)
    if removed:
        log("Disabled SOCKS proxy for arXiv requests")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint filename. If omitted, loads from HF Hub.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--max-results-per-category", type=int, default=2000)
    return parser.parse_args()


def normalize_text(text):
    return " ".join((text or "").split())


def to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def paper_id_from_entry(entry_id):
    return entry_id.rstrip("/").split("/")[-1]


def fetch_recent(days, page_size, delay_seconds, max_results_per_category):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=3)
    client._session.trust_env = False
    client._session.proxies.clear()
    rows = []

    for category in TARGET_CATEGORIES:
        collected = 0
        hit_cap = False
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                published = to_utc(result.published)
                if published < since:
                    break

                categories = list(getattr(result, "categories", []) or [])
                primary_category = getattr(result, "primary_category", None) or (categories[0] if categories else category)
                rows.append({
                    "id": paper_id_from_entry(result.entry_id),
                    "title": normalize_text(result.title),
                    "abstract": normalize_text(result.summary),
                    "categories": " | ".join(categories),
                    "primary_category": primary_category,
                    "published": published.strftime("%Y-%m-%d"),
                    "url": result.entry_id,
                })
                collected += 1
            if collected == max_results_per_category:
                hit_cap = True
            log(f"Fetched {collected} recent papers for {category}")
            if hit_cap:
                log(
                    f"Reached max_results_per_category={max_results_per_category} for {category}; "
                    "recent papers may be truncated"
                )
        except Exception as exc:
            log(f"Skipping {category}: {exc}")

        time.sleep(delay_seconds)

    if not rows:
        return pd.DataFrame(columns=["id", "title", "abstract", "categories", "primary_category", "published", "url"])

    df = (
        pd.DataFrame(rows)
        .sort_values(["published", "id"], ascending=[False, True])
        .drop_duplicates(subset="id", keep="first")
        .reset_index(drop=True)
    )
    log(f"Fetched {len(rows)} rows, kept {len(df)} unique papers")
    return df


def load_model(checkpoint_name=None):
    if checkpoint_name:
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        backbone_src = str(LOCAL_BACKBONE) if LOCAL_BACKBONE.exists() else BASE_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(backbone_src)
        backbone = AutoModel.from_pretrained(backbone_src)
        model = PaperClassifier(backbone).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
    else:
        log(f"Loading model from HF Hub: {MODEL_REPO}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        backbone = AutoModel.from_pretrained(MODEL_REPO)
        model = PaperClassifier(backbone).to(device)
        head_path = hf_hub_download(MODEL_REPO, "classifier_head.pt")
        model.head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    model.eval()
    return tokenizer, model


def push_index_to_hub(output_dir):
    api = HfApi()
    for filename in ["recent_embeddings.npy", "recent_metadata.parquet", "last_updated.txt"]:
        path = output_dir / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=INDEX_REPO, repo_type="dataset")
    log(f"Pushed index to {INDEX_REPO}")


def embed_papers(df, tokenizer, model, batch_size):
    texts = (df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).tolist()
    vectors = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        enc = tokenizer(
            texts[start:end],
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            cls, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        vectors.append(F.normalize(cls, dim=-1).cpu().numpy().astype(np.float32))
        log(f"Embedded {end}/{len(texts)} papers")

    if vectors:
        return np.concatenate(vectors, axis=0)
    return np.empty((0, model.backbone.config.hidden_size), dtype=np.float32)


def save_index(df, embeddings, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "recent_embeddings.npy", embeddings)
    (
        df.assign(
            abstract_preview=df["abstract"].fillna("").map(
                lambda s: (s[:140].rstrip() + "...") if len(s) > 140 else s
            )
        )
        .drop(columns=["abstract"])
        .to_parquet(output_dir / "recent_metadata.parquet", index=False)
    )
    (output_dir / "last_updated.txt").write_text(datetime.now(timezone.utc).isoformat())
    log(f"Saved index to {output_dir}")


def main():
    args = parse_args()
    disable_socks_proxy()
    log(f"Using device: {device}")
    df = fetch_recent(
        days=args.days,
        page_size=args.page_size,
        delay_seconds=args.delay_seconds,
        max_results_per_category=args.max_results_per_category,
    )
    if df.empty:
        raise SystemExit("No recent papers fetched")

    tokenizer, model = load_model(args.checkpoint)
    embeddings = embed_papers(df, tokenizer, model, args.batch_size)
    save_index(df, embeddings, args.output_dir)
    push_index_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
