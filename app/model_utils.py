import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

MODEL_REPO = "AnanasClassic/arxiv-ml-classifier"
INDEX_REPO = "AnanasClassic/arxiv-ml-index"

CATEGORIES = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML", "cs.IR", "cs.RO", "other"]

CATEGORY_LABELS = {
    "cs.LG":   "Machine Learning (cs.LG)",
    "cs.AI":   "Artificial Intelligence (cs.AI)",
    "cs.CV":   "Computer Vision (cs.CV)",
    "cs.CL":   "Computation & Language (cs.CL)",
    "cs.NE":   "Neural & Evolutionary Computing (cs.NE)",
    "stat.ML": "Statistics & Machine Learning (stat.ML)",
    "cs.IR":   "Information Retrieval (cs.IR)",
    "cs.RO":   "Robotics (cs.RO)",
    "other":   "Not ML-related",
}

device = torch.device("cpu")


class PaperClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, len(CATEGORIES))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return cls, self.head(cls)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    backbone = AutoModel.from_pretrained(MODEL_REPO)
    model = PaperClassifier(backbone).to(device)
    head_path = hf_hub_download(MODEL_REPO, "classifier_head.pt")
    model.head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    model.eval()
    return tokenizer, model


def load_recent_index():
    emb_path = hf_hub_download(INDEX_REPO, "recent_embeddings.npy", repo_type="dataset")
    meta_path = hf_hub_download(INDEX_REPO, "recent_metadata.parquet", repo_type="dataset")
    return np.load(emb_path), pd.read_parquet(meta_path)


def _encode(title, abstract, tokenizer):
    text = title + " [SEP] " + (abstract or "")
    return tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")


def classify(title, abstract, tokenizer, model):
    enc = _encode(title, abstract, tokenizer)
    with torch.no_grad():
        _, logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    results, cumsum = [], 0.0
    for i in np.argsort(probs)[::-1]:
        results.append((CATEGORIES[i], float(probs[i])))
        cumsum += probs[i]
        if cumsum >= 0.95:
            break
    return results


def find_similar(title, abstract, tokenizer, model, embeddings, meta, top_k=3):
    enc = _encode(title, abstract, tokenizer)
    with torch.no_grad():
        cls, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    vec = F.normalize(cls, dim=-1).squeeze(0).cpu().numpy()
    scores = embeddings @ vec
    idx = np.argsort(scores)[-top_k:][::-1]
    return meta.iloc[idx].assign(score=scores[idx]).reset_index(drop=True)
