import math
import re
import sys
import time
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

ML_CATS = {"cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML", "cs.IR", "cs.RO"}

ALL_CATS = [
    "astro-ph.CO","astro-ph.EP","astro-ph.GA","astro-ph.HE","astro-ph.IM","astro-ph.SR",
    "cond-mat.dis-nn","cond-mat.mes-hall","cond-mat.mtrl-sci","cond-mat.other",
    "cond-mat.quant-gas","cond-mat.soft","cond-mat.stat-mech","cond-mat.str-el","cond-mat.supr-con",
    "cs.AR","cs.CC","cs.CE","cs.CG","cs.CR","cs.CY","cs.DB","cs.DC","cs.DL","cs.DM",
    "cs.DS","cs.ET","cs.FL","cs.GL","cs.GR","cs.GT","cs.HC","cs.IT","cs.LO","cs.MA",
    "cs.MM","cs.MS","cs.NI","cs.OH","cs.OS","cs.PF","cs.PL","cs.SC","cs.SD","cs.SE","cs.SI",
    "econ.EM","econ.GN","econ.TH",
    "eess.AS","eess.IV","eess.SP","eess.SY",
    "gr-qc","hep-ex","hep-lat","hep-ph","hep-th","nucl-ex","nucl-th","quant-ph",
    "math.AC","math.AG","math.AP","math.AT","math.CA","math.CO","math.CT","math.DG",
    "math.DS","math.FA","math.GM","math.GN","math.GR","math.GT","math.HO","math.KT",
    "math.LO","math.MG","math.NA","math.NT","math.OA","math.OC","math.PR","math.QA",
    "math.RA","math.RT","math.SG","math.SP","math.ST",
    "nlin.AO","nlin.CD","nlin.CG","nlin.PS","nlin.SI",
    "nucl-ex","nucl-th",
    "physics.acc-ph","physics.ao-ph","physics.app-ph","physics.atm-clus","physics.atom-ph",
    "physics.bio-ph","physics.chem-ph","physics.class-ph","physics.comp-ph","physics.data-an",
    "physics.ed-ph","physics.flu-dyn","physics.gen-ph","physics.geo-ph","physics.hist-ph",
    "physics.ins-det","physics.med-ph","physics.optics","physics.plasm-ph","physics.pop-ph",
    "physics.soc-ph","physics.space-ph",
    "q-bio.BM","q-bio.CB","q-bio.GN","q-bio.MN","q-bio.NC","q-bio.OT","q-bio.PE",
    "q-bio.QM","q-bio.SC","q-bio.TO",
    "q-fin.CP","q-fin.GN","q-fin.MF","q-fin.PM","q-fin.PR","q-fin.RM","q-fin.ST","q-fin.TR",
    "stat.AP","stat.CO","stat.ME","stat.OT",
]
NON_ML_CATS = sorted(set(ALL_CATS) - ML_CATS)

TRAIN_TARGET_TOTAL = 10_000
PER_CAT = math.ceil(TRAIN_TARGET_TOTAL / len(NON_ML_CATS))
TEST_YEAR = 2025
TEST_MAX_MONTH = 5
TRAIN_YEARS = range(2024, 1999, -1)
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
    return (
        pl.read_parquet(local, columns=["arxiv_id", "title", "abstract", "submission_date"])
        .rename({"arxiv_id": "id", "submission_date": "published"})
        .with_columns(pl.lit("other").alias("primary_category"))
    )


train_frames = []

MAX_ERRORS = 5

for cat in NON_ML_CATS:
    collected, done, errors = 0, False, 0
    for year in TRAIN_YEARS:
        if done or errors >= MAX_ERRORS:
            break
        for month in range(12, 0, -1):
            if done or errors >= MAX_ERRORS:
                break
            try:
                df = read_file(cat, year, month)
                need = PER_CAT - collected
                if len(df) >= need:
                    df = df.sample(need, seed=42)
                    done = True
                train_frames.append(df)
                collected += len(df)
                errors = 0
                print(f"OK  {cat}/{year}/{month:02d}: +{len(df)} -> {collected}/{PER_CAT}", flush=True)
            except Exception as e:
                errors += 1
                print(f"ERR {cat}/{year}/{month:02d} ({errors}/{MAX_ERRORS}): {e}", flush=True)
    if errors >= MAX_ERRORS:
        print(f"SKIP {cat}: too many errors, collected {collected}/{PER_CAT}", flush=True)


test_frames = []

for cat in NON_ML_CATS:
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
train.write_parquet(DATA_DIR / "papers_train_other.parquet")
print(f"\nTrain other: {len(train)}")

if test_frames:
    test = pl.concat(test_frames)
    test.write_parquet(DATA_DIR / "papers_test_other.parquet")
    print(f"Test other:  {len(test)}")

print(f"\nPer-category target: {PER_CAT} ({len(NON_ML_CATS)} categories)")
