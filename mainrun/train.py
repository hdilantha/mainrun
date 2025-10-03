import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

@dataclass
class Hyperparameters:
    # block_size: int = 128
    block_size: int = 256 # CHANGED
    batch_size: int = 64
    # vocab_size: int = 16_000
    vocab_size: int = 32_000 # CHANGED
    n_layer: int = 6
    # n_head: int = 8
    n_head: int = 12 # CHANGED
    # d_model: int = 512
    d_model: int = 768 # CHANGED
    dropout: float = 0.1
    # lr: float = 6e-3
    lr: float = 1e-3 # CHANGED
    # weight_decay: float = 0.00
    weight_decay: float = 0.01 # CHANGED
    evals_per_epoch: int = 3
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

def precompute_rope_cos_sin(seq_len: int, dim: int, device, base: int = 10000):
    assert dim % 2 == 0, "head_dim must be even for RoPE"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', t, inv_freq)  # (T, half)

    # shape as (1,1,T,half) to broadcast over (B,H,T,half)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, H, T, D) with D even
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]               # (B,H,T,half)

    # Cos/sin already (1,1,T,half); just trim to T
    cosT = cos[..., :T, :]                              # (1,1,T,half)
    sinT = sin[..., :T, :]                              # (1,1,T,half)

    xr_first  = x1 * cosT - x2 * sinT                   # (B,H,T,half)
    xr_second = x1 * sinT + x2 * cosT                   # (B,H,T,half)
    return torch.cat([xr_first, xr_second], dim=-1)     # (B,H,T,D)

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

# class CausalSelfAttention(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         assert cfg.d_model % cfg.n_head == 0
#         self.head_dim = cfg.d_model // cfg.n_head
#         self.n_head   = cfg.n_head
#         self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
#         self.proj = nn.Linear(cfg.d_model, cfg.d_model)
#         self.attn_drop = nn.Dropout(cfg.dropout)
#         self.resid_drop= nn.Dropout(cfg.dropout)
#         self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

#     def forward(self, x: torch.Tensor):
#         B, T, C = x.size()
#         qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
#         q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.resid_drop(self.proj(y))

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.rope_base = 10000
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

    def _maybe_build_rope(self, T: int, device):
        if (self._rope_cos is None) or (self._rope_cos.size(2) < T):
            cos, sin = precompute_rope_cos_sin(T, self.head_dim, device, base=self.rope_base)
            # keep buffers in fp32 is fine; if using AMP, you can cast at use-site:
            self._rope_cos, self._rope_sin = cos, sin


    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        self._maybe_build_rope(T, x.device)

        # qkv: (B,T,3C) -> (B,T,3,H,D)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        # take slices on the 3-axis, then permute to (B,H,T,D)
        q = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # (B,H,T,D)
        k = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()  # (B,H,T,D)
        v = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()  # (B,H,T,D)

        # RoPE on q and k (both (B,H,T,D))
        q = apply_rope(q, self._rope_cos, self._rope_sin)
        k = apply_rope(k, self._rope_cos, self._rope_sin)

        # scaled_dot_product_attention expects (B*H,T,D)
        qh = q.reshape(B * self.n_head, T, self.head_dim)
        kh = k.reshape(B * self.n_head, T, self.head_dim)
        vh = v.reshape(B * self.n_head, T, self.head_dim)

        att_out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )  # (B*H,T,D)

        att_out = att_out.reshape(B, self.n_head, T, self.head_dim).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(att_out))

# class MLP(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(cfg.d_model, 4 * cfg.d_model),
#             nn.GELU(),
#             nn.Linear(4 * cfg.d_model, cfg.d_model),
#             nn.Dropout(cfg.dropout),
#         )
#     def forward(self, x): return self.net(x)

class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Use ~4x d_model capacity via gated path: 2*hidden because gate+value
        hidden = int(2.6667 * cfg.d_model)  # ~= 8/3 d_model (common in LLaMA-ish stacks)
        self.w = nn.Linear(cfg.d_model, hidden, bias=False)
        self.v = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SiLU gate * value
        x = F.silu(self.w(x)) * self.v(x)
        x = self.proj(x)
        return self.drop(x)

# class Block(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(cfg.d_model)
#         self.ln2 = nn.LayerNorm(cfg.d_model)
#         self.attn = CausalSelfAttention(cfg)
#         self.mlp  = MLP(cfg)
#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        # self.mlp  = MLP(cfg)
        self.mlp  = SwiGLU(cfg)
    # def forward(self, x):
    #     x = x + self.attn(self.ln1(x))
    #     x = x + self.mlp(self.ln2(x))
    #     return x
    def forward(self, x):
        y = self.ln1(x)
        a = self.attn(y)
        m = self.mlp(self.ln2(x))  # parallel path from x (NeoX pattern)
        return x + a + m

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.emb_scale = cfg.d_model ** -0.5
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        # pos = self.pos_emb[:, :T, :]
        # x = self.drop(tok + pos)
        tok = self.token_emb(idx) * self.emb_scale
        x = self.drop(tok)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    # opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # CHANGED
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=batches, T_mult=2) # CHANGED

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
