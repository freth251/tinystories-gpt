import os
import time
import numpy as np
import tiktoken
from datasets import load_dataset

CHUNK = 10_000
THREADS = os.cpu_count() or 4

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
assert enc.n_vocab < 2**16


def write_split(split_ds, out_path):
    n_stories = len(split_ds)
    total = 0
    t0 = time.time()

    with open(out_path, "wb") as f:
        for start in range(0, n_stories, CHUNK):
            texts = split_ds[start:start + CHUNK]["text"]
            ids_lists = enc.encode_ordinary_batch(texts, num_threads=THREADS)

            flat = []
            for ids in ids_lists:
                flat.extend(ids)
                flat.append(EOT)

            arr = np.array(flat, dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)

            done = min(start + CHUNK, n_stories)
            if (start // CHUNK) % 20 == 0 or done == n_stories:
                print(f"  {out_path}: {done:,}/{n_stories:,} stories, "
                      f"{total:,} tokens, {time.time() - t0:.0f}s", flush=True)

    size = os.path.getsize(out_path)
    assert size == 2 * total, f"size mismatch: {size} bytes vs {total} tokens"
    print(f"saved {out_path}: {total:,} tokens ({size / 1e6:.0f} MB)", flush=True)
    return total


def check(path, n=300):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"\n--- first {n} tokens of {path} ---")
    print(enc.decode(data[:n].tolist()))
    print("---\n")


if __name__ == "__main__":
    print("loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories")
    print(ds, flush=True)

    write_split(ds["validation"], "val.bin")
    check("val.bin")

    write_split(ds["train"], "train.bin")
    check("train.bin")