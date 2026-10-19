import os
import math
import time
import contextlib
import numpy as np
import torch
from config import (block_size, batch_size, max_iters, eval_interval, eval_iters,
                    grad_accum, warmup_iters, max_lr, min_lr, grad_clip,
                    resume, ckpt_path)
from model import Model

LOG_EVERY = 10

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device:", device, flush=True)

use_amp = device == "cuda"
use_compile = device == "cuda"

if use_amp:
    torch.set_float32_matmul_precision("high")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    ctx = contextlib.nullcontext()


def get_batch(split):
    data = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r")
    idx = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in idx])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(step):
    if step < warmup_iters:
        return max_lr * step / warmup_iters
    ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            total += loss.item()
        out[split] = total / eval_iters
    model.train()
    return out["train"], out["val"]


def save(step):
    torch.save({"model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step}, ckpt_path)


model = Model().to(device)
raw_model = model
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=max_lr)

start_step = 0
if resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

n_params = sum(p.numel() for p in raw_model.parameters())
tokens_per_step = batch_size * block_size * grad_accum
print(f"parameters: {n_params:,} | tokens per step: {tokens_per_step:,}", flush=True)

if use_compile:
    model = torch.compile(model)

t_start = time.time()
t_last = time.time()

for step in range(start_step, max_iters):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if step % eval_interval == 0 or step == max_iters - 1:
        train_l, val_l = estimate_loss()
        mins = (time.time() - t_start) / 60
        print(f"--- step {step:6d} | train {train_l:.4f} | val {val_l:.4f} | {mins:.1f} min ---", flush=True)
        save(step)
        t_last = time.time()

    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        (loss / grad_accum).backward()
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
    optimizer.step()

    if step % LOG_EVERY == 0 and step > start_step:
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t_last) / LOG_EVERY
        t_last = time.time()
        tok_s = tokens_per_step / dt
        eta_h = (max_iters - step) * dt / 3600
        print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | norm {norm:.2f} | "
              f"{dt * 1000:.0f} ms | {tok_s:,.0f} tok/s | eta {eta_h:.1f} h", flush=True)

save(max_iters)
print(f"done. saved {ckpt_path}", flush=True)