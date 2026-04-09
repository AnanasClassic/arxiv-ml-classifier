import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CATEGORIES = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML", "cs.IR", "cs.RO", "other"]
MODEL_NAME = "allenai/specter2_base"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
VAL_CAP = 5_000
VAL_OTHER_CAP = 15_000
VAL_RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)
TRAIN_FILES = ["papers_train_ml.parquet", "papers_train_other.parquet"]
TEST_FILES = ["papers_test_ml.parquet", "papers_test_other.parquet"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


class PaperDataset(Dataset):
    def __init__(self, df, tokenizer, le):
        self.texts = (df["title"] + " [SEP] " + df["abstract"].fillna("")).tolist()
        self.labels = le.transform(df["primary_category"]).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PaperClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, len(CATEGORIES))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return cls, self.head(cls)


def compute_class_weights(df, le):
    counts = df["primary_category"].value_counts()
    weights = np.array([1.0 / counts[cat] for cat in le.classes_])
    weights = weights / weights.sum() * len(CATEGORIES)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            _, logits = model(ids, mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
    return all_labels, all_preds


def load_split(files):
    return pd.concat([pd.read_parquet(DATA_DIR / file_name) for file_name in files], ignore_index=True)


def sample_validation(df):
    def sample_group(category, group):
        limit = VAL_OTHER_CAP if category == "other" else VAL_CAP
        if len(group) <= limit:
            return group
        return group.sample(limit, random_state=VAL_RANDOM_STATE)

    return pd.concat(
        [sample_group(category, group) for category, group in df.groupby("primary_category", sort=False)],
        ignore_index=True,
    )


def resolve_resume_path(resume_arg):
    if not resume_arg:
        return None
    resume_path = Path(resume_arg)
    if not resume_path.exists():
        resume_path = OUT_DIR / resume_arg
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_arg}")
    return resume_path


def move_optimizer_to_device(optimizer):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


args = parse_args()
resume_path = resolve_resume_path(args.resume)


train_df = load_split(TRAIN_FILES)
test_df = sample_validation(load_split(TEST_FILES))
log(f"Train size: {len(train_df)}")
log(f"Validation size: {len(test_df)}")

le = LabelEncoder().fit(CATEGORIES)
json.dump(le.classes_.tolist(), open(OUT_DIR / "label_classes.json", "w"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
backbone = AutoModel.from_pretrained(MODEL_NAME)
model = PaperClassifier(backbone).to(device)

class_weights = compute_class_weights(train_df, le)
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_ds = PaperDataset(train_df, tokenizer, le)
test_ds = PaperDataset(test_df, tokenizer, le)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, num_workers=4, pin_memory=True)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
start_epoch = 0
completed_epochs = []

if resume_path is not None:
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    move_optimizer_to_device(optimizer)
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"]
    completed_epochs.append(start_epoch)
    log(f"Resumed from {resume_path} after epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        _, logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            log(f"Epoch {epoch+1} step {step+1}/{len(train_loader)} loss={total_loss/(step+1):.4f}")

    labels, preds = evaluate(model, test_loader)
    weighted_f1 = f1_score(labels, preds, average="weighted")
    macro_f1 = f1_score(labels, preds, average="macro")
    log(f"Epoch {epoch+1} - weighted F1: {weighted_f1:.4f}  macro F1: {macro_f1:.4f}")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_f1": weighted_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }, OUT_DIR / f"checkpoint_ep{epoch+1}.pt")
    log(f"Saved checkpoint_ep{epoch+1}.pt")
    completed_epochs.append(epoch + 1)

best_ep = max(completed_epochs, key=lambda e: torch.load(OUT_DIR / f"checkpoint_ep{e}.pt", weights_only=True)["val_f1"])
log(f"Best epoch by weighted F1: {best_ep}")

ckpt = torch.load(OUT_DIR / f"checkpoint_ep{best_ep}.pt", weights_only=True)
model.load_state_dict(ckpt["model"])
labels, preds = evaluate(model, test_loader)
print(classification_report(labels, preds, target_names=le.classes_))

backbone.save_pretrained(OUT_DIR / "backbone")
tokenizer.save_pretrained(OUT_DIR / "backbone")
torch.save(model.head.state_dict(), OUT_DIR / "classifier_head.pt")
log(f"Done. Best epoch {best_ep} saved to {OUT_DIR}")
