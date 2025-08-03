import os
import numpy as np
from datasets import load_dataset
import tiktoken
import pickle

dataset = load_dataset("roneneldan/TinyStories")

start_of_story = "<|startofstory|>\n"
end_of_story = "\n<|endofstory|>\n"

enc = tiktoken.get_encoding("gpt2")

def compute_token_stats(data, encoder):
    token_lengths = []
    for story in data:
        tokens = encoder.encode_ordinary(story)
        token_lengths.append(len(tokens))
    avg_tokens = np.mean(token_lengths)
    max_tokens = np.max(token_lengths)
    min_tokens = np.min(token_lengths)
    return avg_tokens, max_tokens, min_tokens, token_lengths

train_data = [f"{start_of_story}{story}{end_of_story}" for story in dataset["train"]["text"]]
val_data = [f"{start_of_story}{story}{end_of_story}" for story in dataset["validation"]["text"]]

print("Statistical training set token information...")
train_avg, train_max, train_min, train_lengths = compute_token_stats(train_data, enc)
print(f"Training set - average number of tokens: {train_avg:.2f}, Maximum number of tokens: {train_max}, Minimum number of tokens: {train_min}")

print("Statistical validation set token information...")
val_avg, val_max, val_min, val_lengths = compute_token_stats(val_data, enc)
print(f"Validation set - average number of tokens:{val_avg:.2f}, Maximum number of tokens:{val_max}, Minimum number of tokens: {val_min}")

train_data_str = "".join(train_data)
val_data_str = "".join(val_data)

train_ids = enc.encode_ordinary(train_data_str)
val_ids = enc.encode_ordinary(val_data_str)

print(f"The training set contains {len(train_ids):,} tokens")
print(f"The validation set contains {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

output_dir = os.path.dirname(__file__)
train_bin_path = os.path.join(output_dir, "train.bin")
val_bin_path = os.path.join(output_dir, "val.bin")

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)

print(f"The training set is saved to {train_bin_path}")
print(f"The validation set is saved to {val_bin_path}")

vocab_size = enc.n_vocab
meta = {
    'vocab_size': vocab_size,
    'start_of_story': start_of_story,
    'end_of_story': end_of_story
}
meta_path = os.path.join(output_dir, "meta.pkl")
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Metadata is saved to {meta_path}")