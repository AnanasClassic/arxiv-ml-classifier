import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "checkpoint_ep3.pt"
HF_REPO = "AnanasClassic/arxiv-ml-classifier"
CATEGORIES = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML", "cs.IR", "cs.RO", "other"]
OUT_DIR = Path(__file__).parent / "checkpoints"
EXPORT_DIR = Path(__file__).parent / "export"
EXPORT_DIR.mkdir(exist_ok=True)


class PaperClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, len(CATEGORIES))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return cls, self.head(cls)


print(f"Loading {CHECKPOINT}...")
ckpt = torch.load(OUT_DIR / CHECKPOINT, map_location="cpu", weights_only=True)

backbone_src = str(OUT_DIR / "backbone") if (OUT_DIR / "backbone").exists() else "allenai/specter2_base"
tokenizer = AutoTokenizer.from_pretrained(backbone_src)
backbone = AutoModel.from_pretrained(backbone_src)
model = PaperClassifier(backbone)
model.load_state_dict(ckpt["model"])
model.eval()

print("Saving export artifacts...")
model.backbone.save_pretrained(EXPORT_DIR)
tokenizer.save_pretrained(EXPORT_DIR)
torch.save(model.head.state_dict(), EXPORT_DIR / "classifier_head.pt")
json.dump(CATEGORIES, open(EXPORT_DIR / "label_classes.json", "w"))

print(f"Pushing to {HF_REPO}...")
api = HfApi()
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(EXPORT_DIR), repo_id=HF_REPO, repo_type="model")

print("Done.")
print(f"  macro F1 ep3: {ckpt.get('macro_f1', 'n/a')}")
print(f"  weighted F1:  {ckpt.get('weighted_f1', 'n/a')}")
