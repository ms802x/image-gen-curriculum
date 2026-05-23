"""Fetch lambdalabs/naruto-blip-captions, save 10 images to data/raw/ and full set to data/full/."""
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FULL = ROOT / "data" / "full"
RAW.mkdir(parents=True, exist_ok=True)
FULL.mkdir(parents=True, exist_ok=True)

ds = load_dataset("lambdalabs/naruto-blip-captions", split="train")
print(f"Loaded: {len(ds)} examples; features = {ds.features}")

random.seed(0)
indices = random.sample(range(len(ds)), 10)

raw_captions = {}
for slot, idx in enumerate(indices):
    ex = ds[idx]
    img = ex["image"].convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512))
    fname = f"{slot:02d}.png"
    img.save(RAW / fname)
    raw_captions[fname] = ex["text"]
    print(f"  raw/{fname}  size={img.size}  caption={ex['text'][:80]}")

(RAW / "captions.json").write_text(json.dumps(raw_captions, indent=2))

full_captions = {}
for i, ex in enumerate(ds):
    img = ex["image"].convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512))
    fname = f"{i:05d}.png"
    img.save(FULL / fname)
    full_captions[fname] = ex["text"]

(FULL / "captions.json").write_text(json.dumps(full_captions, indent=2))
print(f"\nWrote {len(raw_captions)} to {RAW}")
print(f"Wrote {len(full_captions)} to {FULL}")
