#!/usr/bin/env python
# prepare.py

import os
import pickle
import random

import numpy as np
import tiktoken
from datasets import load_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

PARA_TOKEN = "<|PARA|>"
EOT_TOKEN  = ""

enc = tiktoken.get_encoding("gpt2")
para_id = enc.encode(PARA_TOKEN)
eot_id  = enc.encode(EOT_TOKEN)

print(f"Paragraph token IDs: {para_id}, End-of-text token IDs: {eot_id}")
print(f"Writing to directory: {OUT_DIR}")

def tokenize_story(text: str) -> list[int]:
    """Encode a multi-paragraph story: add PARA_TOKEN after each paragraph and EOT_TOKEN at the end."""
    toks: list[int] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        toks.extend(enc.encode(para))
        toks.extend(para_id)
    toks.extend(eot_id)
    return toks

# -----------------------------------------------------------------------------
def build_train_val(train_split, val_split, train_file="train.bin", val_file="val.bin", val_ratio=0.1):
    """
    - Split train_split into a 90/10 split, first writing it to train.bin & val_temp.bin
    - Then, encode the entire HF validation_split and append it to val.bin
    """
    train_tokens = []
    val_tokens   = []

    for ex in train_split:
        toks = tokenize_story(ex['text'])
        if random.random() < val_ratio:
            val_tokens.extend(toks)
        else:
            train_tokens.extend(toks)

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_path = os.path.join(OUT_DIR, train_file)
    train_arr.tofile(train_path)
    print(f"Wrote {train_path}: {len(train_arr):,} tokens")

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_path = os.path.join(OUT_DIR, val_file)
    val_arr.tofile(val_path)
    print(f"Wrote {val_path} (from train split): {len(val_arr):,} tokens")

    append_tokens = []
    for ex in val_split:
        append_tokens.extend(tokenize_story(ex['text']))

    all_val = np.concatenate([val_arr, np.array(append_tokens, dtype=np.uint16)])
    all_val.tofile(val_path)
    print(f"Appended HF validation → {val_path}: now {len(all_val):,} tokens")


if __name__ == "__main__":
    random.seed(42)

    ds = load_dataset("roneneldan/TinyStories")
    print(f"Train samples:      {len(ds['train']):,}")
    print(f"Validation samples: {len(ds['validation']):,}")

    build_train_val(
        train_split=ds['train'],
        val_split=ds['validation'],
        train_file="train.bin",
        val_file="val.bin",
        val_ratio=0.1
    )

    meta = {
        'vocab_size': enc.n_vocab,
        'para_token': para_id,
        'eot_token': eot_id,
    }
    meta_path = os.path.join(OUT_DIR, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote {meta_path}: {meta}")
