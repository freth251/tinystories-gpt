import sys
import torch
from model import Model
import tiktoken

torch.manual_seed(42)        # change this number to get a different sample

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

enc = tiktoken.get_encoding("gpt2")

def encode(s):
    return enc.encode(s)

def decode(ids):
    return enc.decode(ids)

# ---- settings from the command line ----
n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 500
start = sys.argv[2] if len(sys.argv) > 2 else "\n"

# ---- load the model ----
model = Model()
state = torch.load("ckpt.pt", map_location=device)   # map_location: works on any machine
model.load_state_dict(state["model"])
model = model.to(device)
model.eval()                                          # turns dropout off

# ---- generate ----
idx = torch.tensor([encode(start)], dtype=torch.long, device=device)   # shape (1, T)
with torch.no_grad():
    out = model.generate(idx, n_new)

print(decode(out[0].tolist()))