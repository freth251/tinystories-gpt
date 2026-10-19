import torch
from torch import nn
import torch.nn.functional as F
from config import n_embd, dropout, block_size, n_head, n_layer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
    def forward(self, x): 
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs = [h(x) for h in self.heads] 
        outs = torch.cat(outs, dim=-1)
        outs = self.proj(outs)
        outs = self.dropout(outs)
        return outs
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq= nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.seq(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.att = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        out = x + self.att(self.ln1(x))
        out = out + self.ffwd(self.ln2(out))
        return out 

class Model(nn.Module):
        def __init__(self, vocab_size = 50257):
                super().__init__()
                self.table = nn.Embedding(vocab_size, n_embd)
                self.lm_head = nn.Linear(n_embd, vocab_size)
                self.pos_table = nn.Embedding(block_size, n_embd)
                self.blocks = nn.Sequential(*[Block(n_head) for _ in range(n_layer)])
                self.ln = nn.LayerNorm(n_embd)
                self.dropout = nn.Dropout(dropout)
                self.lm_head.weight = self.table.weight
                self.apply(self._init_weights)

        def forward(self, idx, targets=None):
                B, T = idx.shape
                tok = self.table(idx)
                pos = self.pos_table(torch.arange(T, device=idx.device))
                x = pos + tok
                x = self.blocks(x)
                x = self.ln(x)
                logits = self.lm_head(x)
                if targets == None: 
                        return logits, None
                else:   
                        _, _, C = logits.shape
                        loss =  F.cross_entropy(logits.view(-1, C), targets.view(-1)) 
                        return logits, loss
        def generate(self, idx, max_new_tokens, block_size = block_size):
                for _ in range(max_new_tokens): 
                        logits, _ = self.forward(idx[:, -block_size:])
                        logits = logits[:, -1, :]
                        p = F.softmax(logits, dim=-1)
                        pred = torch.multinomial(p, num_samples=1)
                        idx = torch.cat((idx, pred), dim=1)
                return idx
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)