import re
import sys
import time
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

CATS = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML", "cs.IR", "cs.RO"]
TRAIN_TARGET = 10_000
TRAIN_YEARS = range(2024, 1999, -1)
TEST_YEAR = 2025
TEST_MAX_MONTH = 9 # за 10.25 уже нет статей
MAX_ERRORS = 5
MONTHS = range(12, 0, -1)
DATA_DIR = Path(__file__).parent
CAT_CODE_RE = re.compile(r"\((\S+)\)")


def download(cat, year, month, retries=3):
    path = f"data/{cat}/{year}/{month:02d}/00000000.parquet"
    for attempt in range(retries):
        try:
            return hf_hub_download("permutans/arxiv-papers-by-subject", path, repo_type="dataset")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise e


def read_file(cat, year, month):
    local = download(cat, year, month)
    df = pl.read_parquet(local, columns=["arxiv_id", "title", "abstract", "primary_subject", "submission_date"])
    return df.rename({"arxiv_id": "id", "submission_date": "published"}).with_columns(
        pl.col("primary_subject")
        .map_elements(lambda s: (m := CAT_CODE_RE.search(s or "")) and m.group(1), return_dtype=pl.String)
        .alias("primary_category")
    ).filter(pl.col("primary_category") == cat).select(["id", "title", "abstract", "primary_category", "published"])


train_frames = []
earliest = {}

for cat in CATS:
    frames, total = [], 0
    for year in TRAIN_YEARS:
        if total >= TRAIN_TARGET:
            break
        for month in MONTHS:
            if total >= TRAIN_TARGET:
                break
            try:
                df = read_file(cat, year, month)
                need = TRAIN_TARGET - total
                if len(df) > need:
                    df = df.sample(need, seed=42)
                frames.append(df)
                total += len(df)
                earliest[cat] = (year, month)
                print(f"OK  {cat}/{year}/{month:02d}: +{len(df)} -> {total}", flush=True)
            except Exception as e:
                print(f"ERR {cat}/{year}/{month:02d}: {e}", flush=True)
    if frames:
        train_frames.append(pl.concat(frames))
    print(f"  {cat}: {total} train papers, earliest {earliest.get(cat)}", flush=True)

test_frames = []

for cat in CATS:
    errors = 0
    for month in range(1, TEST_MAX_MONTH + 1):
        if errors >= MAX_ERRORS:
            break
        try:
            df = read_file(cat, TEST_YEAR, month)
            test_frames.append(df)
            errors = 0
            print(f"TEST {cat}/2025/{month:02d}: {len(df)}", flush=True)
        except Exception as e:
            errors += 1
            print(f"ERR  {cat}/2025/{month:02d} ({errors}/{MAX_ERRORS}): {e}", flush=True)

if not train_frames:
    print("No train data", file=sys.stderr)
    sys.exit(1)

train = pl.concat(train_frames)
train.write_parquet(DATA_DIR / "papers_train_ml.parquet")
print(f"\nTrain: {len(train)}")
print(train["primary_category"].value_counts().sort("primary_category"))

if test_frames:
    test = pl.concat(test_frames)
    test.write_parquet(DATA_DIR / "papers_test_ml.parquet")
    print(f"\nTest: {len(test)}")
    print(test["primary_category"].value_counts().sort("primary_category"))

print("\n=== Earliest year/month per category ===")
for cat, (y, m) in sorted(earliest.items()):
    print(f"  {cat}: {y}/{m:02d}")
