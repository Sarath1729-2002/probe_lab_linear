"""Shared pieces: hidden-state reader, answer matching, probe fitting, grouped CV."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np

DEVICE = os.environ.get("ERRPROP_DEVICE", "cuda")
NUM = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def run_dir(cfg: dict, tag: str) -> Path:
    d = Path(os.path.expandvars(cfg["out_dir"])) / f"{cfg['run_prefix']}-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(cfg: dict) -> Path:
    d = Path(os.path.expandvars(cfg["data_dir"]))
    d.mkdir(parents=True, exist_ok=True)
    return d


def layer_list(n_layers: int, step: int) -> list[int]:
    return sorted(set(range(0, n_layers, step)) | {n_layers - 1})


# ----------------------------------------------------------------------------- answers

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_match(answer: str, aliases: list[str]) -> bool:
    """True if any alias appears in the answer as whole words (after normalisation)."""
    a = f" {norm(answer)} "
    for al in aliases:
        n = norm(al)
        if len(n) >= 2 and f" {n} " in a:
            return True
    return False


def last_number(text: str) -> float | None:
    if text is None:
        return None
    m = re.findall(r"####\s*\$?\s*(" + NUM.pattern + ")", text)
    nums = m or NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def num_equal(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ----------------------------------------------------------------------------- generation

def chat(tok, user: str, no_system: bool = False) -> str:
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_generator(cfg: dict, model_name: str):
    from errprop.gen import Generator
    g = {"backend": cfg["gen"]["backend"], "max_model_len": cfg["gen"]["max_model_len"],
         "gpu_mem": cfg["gen"]["gpu_mem"], "top_p": 1.0}
    return Generator({"model": model_name, "seed": cfg["seed"], "gen": g})


# ----------------------------------------------------------------------------- hidden states

class Reader:
    """Frozen model; reads residual-stream states at chosen token positions for a batch of texts."""

    def __init__(self, model_name: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=DEVICE).eval()
        self.n_layers = self.model.config.num_hidden_layers + 1
        self.d = self.model.config.hidden_size

    def read(self, texts: list[str], layers: list[int], char_ends: list[list[int]] | None = None,
             add_special: bool = False, bs: int = 16):
        """Returns (H, nll): H[i, p, l, :] = state after the token that ends at char position
        char_ends[i][p] (or the last token when char_ends is None), at layer layers[l];
        nll[i] = mean per-token negative log-likelihood of text i (tokens after the first)."""
        torch = self.torch
        P = 1 if char_ends is None else len(char_ends[0])
        H = np.zeros((len(texts), P, len(layers), self.d), np.float16)
        nll = np.zeros(len(texts), np.float32)
        for s in range(0, len(texts), bs):
            batch = texts[s:s + bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=add_special,
                           return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l] for l in layers], 1)          # B, L, T, D
            lens = enc["attention_mask"].sum(1)
            logp = torch.log_softmax(out.logits.float(), -1)
            for j in range(len(batch)):
                n = int(lens[j])
                tgt = enc["input_ids"][j, 1:n]
                lp = logp[j, : n - 1].gather(1, tgt[:, None]).squeeze(1)
                nll[s + j] = float(-lp.mean()) if n > 1 else 0.0
                if char_ends is None:
                    idx = [n - 1]
                else:
                    o = offs[j][:n]
                    idx = []
                    for ce in char_ends[s + j]:
                        cand = [t for t in range(n) if o[t][1] <= ce and o[t][1] > o[t][0]]
                        idx.append(cand[-1] if cand else n - 1)
                for p, t in enumerate(idx):
                    H[s + j, p] = hs[j, :, t].float().cpu().numpy().astype(np.float16)
        return H, nll


# ----------------------------------------------------------------------------- probes

def logistic(C: float = 0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


class MassMean:
    """Difference-of-means direction (Marks & Tegmark); score = projection onto it."""

    def fit(self, X, y):
        X = np.asarray(X, np.float32)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
        return self

    def decision_function(self, X):
        return ((np.asarray(X, np.float32) - self.mu) / self.sd) @ self.w

    @property
    def direction(self):
        return self.w / (np.linalg.norm(self.w) + 1e-12)


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def group_split(groups, test_frac: float, seed: int):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    test_g = set(rng.choice(uniq, max(1, int(round(test_frac * len(uniq)))), replace=False))
    te = np.array([g in test_g for g in groups])
    return ~te, te


def best_layer(X, y, groups, make, folds: int = 3) -> tuple[int, list[float]]:
    """Layer chosen by grouped CV AUROC on the training rows. X: [N, L, D]."""
    from sklearn.model_selection import GroupKFold
    nf = min(folds, len(np.unique(groups)))
    curve = []
    for l in range(X.shape[1]):
        Xl = X[:, l].astype(np.float32)
        oof = np.zeros(len(y))
        ok = True
        for tr, va in GroupKFold(nf).split(Xl, y, groups):
            if len(np.unique(y[tr])) < 2:
                ok = False
                break
            oof[va] = make().fit(Xl[tr], y[tr]).decision_function(Xl[va])
        curve.append(auc(y, oof) if ok else float("nan"))
    c = np.nan_to_num(np.array(curve), nan=-1)
    return int(np.argmax(c)), curve


def oof_scores(X, y, groups, make, folds: int = 5):
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(min(folds, len(np.unique(groups)))).split(X, y, groups):
        if len(np.unique(y[tr])) == 2:
            oof[va] = make().fit(X[tr], y[tr]).decision_function(X[va])
    return oof


def boot_ci(fn, groups, n: int = 500, seed: int = 0):
    """95% interval of fn(index array) under resampling of whole groups."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        v = fn(idx)
        if v == v:
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def coef_direction(model) -> np.ndarray:
    """Unit direction of a fitted probe in standardised space."""
    if isinstance(model, MassMean):
        return model.direction
    w = model[-1].coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)
