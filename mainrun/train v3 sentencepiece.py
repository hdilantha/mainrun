import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

import sentencepiece as spm
import os


import re
import unicodedata

# Map smart punctuation to ASCII
SMARTS = {
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-",
    "\u00A0": " ",
}
SMART_RE = re.compile("|".join(map(re.escape, SMARTS.keys())))

ASK_RE  = re.compile(r"^\s*Ask HN:\s*",  flags=re.IGNORECASE)
SHOW_RE = re.compile(r"^\s*Show HN:\s*", flags=re.IGNORECASE)
TELL_RE = re.compile(r"^\s*Tell HN:\s*", flags=re.IGNORECASE)
URL_RE  = re.compile(r"https?://\S+")

def preprocess_title(t: str) -> str:
    if not t: return ""
    t = unicodedata.normalize("NFKC", t)
    t = SMART_RE.sub(lambda m: SMARTS[m.group(0)], t)
    t = URL_RE.sub("<URL>", t)
    if ASK_RE.match(t):  t = ASK_RE.sub("<ASK> ", t)
    if SHOW_RE.match(t): t = SHOW_RE.sub("<SHOW> ", t)
    if TELL_RE.match(t): t = TELL_RE.sub("<TELL> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class Hyperparameters:
    # block_size: int = 128
    block_size: int = 256 # CHANGED
    batch_size: int = 64
    # vocab_size: int = 16_000
    vocab_size: int = 32_000 # CHANGED
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-3
    # lr: float = 1e-3 # CHANGED
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

# def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
#     tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
#     tokenizer.decoder = decoders.ByteLevel()
#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=[pad_token, eos_token, unk_token]
#     )
#     tokenizer.train_from_iterator(titles, trainer)
#     return tokenizer

# class BPETokenizer:
#     def __init__(self, tokenizer: Tokenizer):
#         self.tk = tokenizer
#         self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
#         self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

#     def encode(self, s: str) -> list[int]:
#         return self.tk.encode(s).ids

#     def decode(self, ids: list[int]) -> str:
#         return self.tk.decode(ids, skip_special_tokens=True)

#     @property
#     def vocab_size(self): return self.tk.get_vocab_size()

def train_sentencepiece(titles: list[str], vocab_size: int, model_prefix: str) -> str:
    import os, sentencepiece as spm
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    tmp_txt = f"{model_prefix}.train.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        for t in titles:
            tt = (t or "").strip()
            if tt:
                f.write(tt + "\n")

    # URL/techy fragments commonly found in HN titles.
    # (No spaces; these match verbatim substrings in your raw data.)
    user_syms = (
        "<eos>", "<pad>",
        "http", "https", "://", "www.", ".com", ".org", ".io",
        ".dev", ".ai", ".net", ".gg",
        "C++", "C#", "API", "SDK", "GPU", "CLI",
        "HTTP", "HTTPS", "SQL", "NoSQL",
        "macOS", "iOS", "Android", "Linux", "Unix",
        "RFC", "YC", "OSS",
        "GPT-4", "GPT-4o", "LLM", "AI"
    )

    spm.SentencePieceTrainer.train(
        input=tmp_txt,
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="nfkc",  # safe normalization, not augmentation
        byte_fallback=True,
        remove_extra_whitespaces=True,
        split_by_number=False,           # keep 2024 / 1.2.3 tighter
        hard_vocab_limit=False,
        user_defined_symbols=",".join(user_syms),
        input_sentence_size=4000000,
        shuffle_input_sentence=True,
        num_threads=0
    )
    return f"{model_prefix}.model"



class SPMTokenizer:
    def __init__(self, model_file: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(model_file):
            raise RuntimeError(f"Failed to load SentencePiece model: {model_file}")
        self.eos_id = self.sp.piece_to_id("<eos>")
        assert self.eos_id != self.sp.unk_id(), "<eos> must be in vocab"

    def encode_ids(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)  # deterministic

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())


def encode_titles_to_flat_ids(titles: list[str], tok: SPMTokenizer,
                              sampling: bool, alpha: float) -> torch.LongTensor:
    out = []
    for t in titles:
        tt = preprocess_title(t)
        if not tt: 
            continue
        out.extend(tok.encode_ids(tt, sampling=sampling, alpha=alpha, nbest=-1))
        out.append(tok.eos_id)  # boundary
    return torch.tensor(out, dtype=torch.long)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
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
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
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
    
    # eos_token = "<eos>"
    # tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    # train_text = eos_token.join(train_titles) + eos_token
    # val_text = eos_token.join(val_titles) + eos_token
    # train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    # val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    eos_token = "<eos>"
    spm_prefix = f"./data/hn_spm_{args.vocab_size}"
    spm_model_path = f"{spm_prefix}.model"
    if not os.path.exists(spm_model_path):
        spm_model_path = train_sentencepiece(train_titles + val_titles, args.vocab_size, spm_prefix)

    tok = SPMTokenizer(spm_model_path)

    train_text = eos_token.join(train_titles) + eos_token
    val_text   = eos_token.join(val_titles) + eos_token
    train_ids  = torch.tensor(tok.encode_ids(train_text), dtype=torch.long)
    val_ids    = torch.tensor(tok.encode_ids(val_text),   dtype=torch.long)

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
