import os
import time
import pickle
import numpy as np
import gc
from datasets import load_dataset


class ProgressTracker:
    def __init__(self, total_stories):
        self.start_time = time.time() + 1e-9
        self.total = int(total_stories)
        self.count = 0
        self.last_print = 0
        self.rate_history = []

    def update(self, batch=1):
        self.count = min(self.count + batch, self.total)
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)

        current_rate = self.count / elapsed
        self.rate_history.append(current_rate)
        if len(self.rate_history) > 5:
            self.rate_history.pop(0)
        rate = np.median(self.rate_history)

        remaining = max((self.total - self.count) / rate, 0) if rate > 0 else 0

        if now - self.last_print > 5:
            print(f"Processed {self.count}/{self.total} ({self.progress:.1%}) | "
                  f"Elapsed: {self.format_time(elapsed)} | "
                  f"ETA: {self.format_time(remaining)}")
            self.last_print = now

    @property
    def progress(self):
        return min(self.count / self.total, 1.0) if self.total > 0 else 0.0

    @staticmethod
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        mins, secs = divmod(seconds, 60)
        if mins < 60:
            return f"{mins:.0f}m {secs:.0f}s"
        hours, mins = divmod(mins, 60)
        return f"{hours:.0f}h {mins:.0f}m"


def prepare_tinystories():
    try:
        print("Loading dataset...")
        dataset = load_dataset("roneneldan/TinyStories")

        # Validation dataset structure
        required_splits = {'train', 'validation'}
        assert required_splits.issubset(dataset), f"Lack of necessary data segmentation: {required_splits}"
        train_size = len(dataset['train'])
        val_size = len(dataset['validation'])
        print(f"Dataset Statistics: Training Set {train_size} Stories | Validation Set {val_size} Stories")

        print("\n Collect the character list...")
        chars = set()

        print("[Training set character collection]")
        train_tracker = ProgressTracker(train_size)
        for example in dataset['train']:
            chars.update(example['text'])
            train_tracker.update()
            del example
            if train_tracker.count % 1000 == 0:
                gc.collect()

        print("\n[Verification set character collection]")
        val_tracker = ProgressTracker(val_size)
        for example in dataset['validation']:
            chars.update(example['text'])
            val_tracker.update()
            del example
            if val_tracker.count % 1000 == 0:
                gc.collect()

        # Verification character table
        if not chars:
            raise ValueError("No valid characters were collected")
        chars = sorted(chars)
        vocab_size = len(chars)
        print(f"\n Character table collection completed | Size: {vocab_size} | Example: {''.join(chars[:20])}...")

        print("\n Start encoding and saving data...")
        stoi = {ch: i for i, ch in enumerate(chars)}

        def process_split(split_name, expected_size):
            output_file = "train.bin" if split_name == "train" else "val.bin"
            print(f"\n Processing {split_name} -> {output_file}")

            total_tokens = 0
            tracker = ProgressTracker(expected_size)

            try:
                with open(output_file, 'wb') as f:
                    for example in dataset[split_name]:
                        text = example['text'] + '\n'  # Add story dividers
                        encoded = np.array([stoi[c] for c in text], dtype=np.uint16)

                        f.write(encoded.tobytes())
                        total_tokens += len(encoded)
                        tracker.update()

                        del example, text, encoded
                        if tracker.count % 500 == 0:
                            gc.collect()

                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"document{output_file}Not generated")
                if os.path.getsize(output_file) == 0:
                    raise RuntimeError(f"document{output_file}Empty")

                print(f" {split_name} Processing completed | Tokens: {total_tokens:,}")
                return total_tokens
            except Exception as e:
                if os.path.exists(output_file):
                    os.remove(output_file)
                raise RuntimeError(f"{split_name} Processing failure") from e

        train_tokens = process_split('train', train_size)
        val_tokens = process_split('validation', val_size)

        meta = {
            'vocab_size': vocab_size,
            'itos': {i: ch for i, ch in enumerate(chars)},
            'stoi': stoi,
            'dataset_stats': {
                'train_stories': train_size,
                'val_stories': val_size,
                'train_tokens': train_tokens,
                'val_tokens': val_tokens
            }
        }
        with open('meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

        print("\n Output file verification:")
        for fname in ['train.bin', 'val.bin', 'meta.pkl']:
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Key documents{fname}Missing")
            size_mb = os.path.getsize(fname) / (1024 ** 2)
            print(f"  {fname}: {size_mb:.1f}MB")

        print("\n Data processing is complete!")

    except Exception as e:
        print(f"\n Handling Exceptions: {str(e)}")
        raise


if __name__ == "__main__":
    prepare_tinystories()