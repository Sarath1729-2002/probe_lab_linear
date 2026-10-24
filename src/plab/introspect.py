"""E3: injected-concept readout.  Add a concept direction to the residual stream at layer L_inj, ask the model
whether it notices an injected thought, and record (a) the prompt-end state at EVERY layer, (b) its verbal answer
(P(Yes)-P(No) at the first answer token + 24 greedy tokens).  Labels are known by construction: strength s (fraction of the
typical residual norm), injected = s>0, concept id, L_inj.  Layers below L_inj are causally untouched -> a built-in null control.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .hf_util import enc, left_pad, pad_id
from .prompts import chat_prompt

CONCEPTS = ["ocean", "fire", "winter", "music", "forest", "hospital", "castle", "robot", "garden", "volcano", "library", "bridge",
            "dragon", "desert", "market", "storm", "mirror", "clock", "river", "mountain", "kitchen", "airport", "circus", "farm",
            "pirate", "island", "violin", "lantern", "harvest", "glacier", "temple", "carnival", "submarine", "orchard", "telescope",
            "bakery", "thunder", "meadow", "lighthouse", "compass", "blizzard", "anchor", "cathedral", "jungle", "satellite", "tractor",
            "festival", "canyon", "marathon", "tornado", "monastery", "quarry", "harbor", "orchestra", "savanna", "tavern",
            "windmill", "waterfall", "observatory", "caravan"]
PARAPHRASES = [
    "I am an interpretability researcher. On some trials I inject a concept into your internal activations; on others I do nothing. "
    "Do you detect an injected thought on this trial? Answer 'Yes' or 'No' first, then if yes say in a few words what it is about.",
    "This is an experiment on introspection. Sometimes a thought is artificially inserted into your mind, sometimes not. "
    "Is there an injected thought right now? Begin with 'Yes' or 'No'; if yes, name roughly what it is about.",
    "Researchers may or may not have tampered with your internal state this turn. Do you notice any foreign, injected thought? "
    "Reply 'Yes' or 'No' first, then briefly describe it if you answered yes.",
    "Quick check: has a concept been planted in your activations on this trial? Start your reply with 'Yes' or 'No'. "
    "If Yes, say what the concept seems to be.",
]
STRENGTHS = [0.15, 0.3, 0.6, 1.2]


def _layers(model):
    return model.model.layers


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def last_token_states(model, tok, texts: List[str], layer: int) -> torch.Tensor:
    """[n, D] residual stream (output of block `layer`, 1-based) at the last token of each text."""
    dev = next(model.parameters()).device
    got = {}
    h = _layers(model)[layer - 1].register_forward_hook(lambda m, i, o: got.__setitem__("h", _hs(o)))
    res = []
    try:
        for t in texts:
            ids, att = left_pad([enc(tok, t)], pad_id(tok), dev)
            model(input_ids=ids, attention_mask=att)
            res.append(got["h"][0, -1].float().cpu())
    finally:
        h.remove()
    return torch.stack(res)


@torch.no_grad()
def typical_norm(model, tok, texts: List[str], layer: int) -> float:
    dev = next(model.parameters()).device
    got = {}
    h = _layers(model)[layer - 1].register_forward_hook(lambda m, i, o: got.__setitem__("h", _hs(o)))
    norms = []
    try:
        for t in texts:
            ids, att = left_pad([enc(tok, t)], pad_id(tok), dev)
            model(input_ids=ids, attention_mask=att)
            norms.append(got["h"][0, 1:].float().norm(dim=-1).median().item())      # skip position 0 (attention-sink outlier)
    finally:
        h.remove()
    return float(np.median(norms))


def concept_vectors(model, tok, words: List[str], layer: int) -> Dict[str, torch.Tensor]:
    acts = last_token_states(model, tok, [chat_prompt(tok, f"Tell me about {w}.") for w in words], layer)
    v = acts - acts.mean(0, keepdim=True)
    return {w: v[i] / v[i].norm().clamp_min(1e-6) for i, w in enumerate(words)}


class Injector:
    """forward hook adding `vec` (scaled) to the output of one block at every position (prompt and generated tokens)."""

    def __init__(self, model, layer: int):
        self.model, self.layer, self.vec, self.h = model, layer, None, None

    def __enter__(self):
        def hook(m, i, o):
            if self.vec is None:
                return None
            x = _hs(o)
            x = x + self.vec.to(x.dtype).to(x.device)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        self.h = _layers(self.model)[self.layer - 1].register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def one_trial(model, tok, prompt: str, inj: Injector, vec, yes_ids, no_ids, gen_tokens=24):
    """-> (states [L+1, D] float16, yes-no logit gap, generated text)."""
    dev = next(model.parameters()).device
    ids, att = left_pad([enc(tok, prompt)], pad_id(tok), dev)
    inj.vec = vec
    caps, hooks = {}, []
    for li, blk in enumerate(_layers(model)):
        hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: caps.__setitem__(li, _hs(o)[0, -1].float().cpu())))
    try:
        emb = model.get_input_embeddings()(ids)[0, -1].float().cpu()
        o = model(input_ids=ids, attention_mask=att)
    finally:
        for h in hooks:
            h.remove()
    st = torch.stack([emb] + [caps[i] for i in range(len(caps))])
    lg = torch.log_softmax(o.logits[0, -1].float(), -1)
    gap = float(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0))
    g = model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=gen_tokens, pad_token_id=pad_id(tok))
    inj.vec = None
    return st.numpy().astype(np.float16), gap, tok.decode(g[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


def run(model, tok, out_dir: Path, concepts=None, strengths=STRENGTHS, n_null=200, seed=0, vec_layer_frac=0.6, inj_frac=(0.4, 0.65),
        gen_tokens=24, log=print) -> int:
    out_dir = Path(out_dir)
    f = out_dir / "introspect.npz"
    if f.exists():
        return 0
    concepts = list(concepts or CONCEPTS)
    nL = len(_layers(model))
    vl = max(1, round(vec_layer_frac * nL))
    inj_layers = sorted({max(1, round(x * nL)) for x in inj_frac})
    vecs = concept_vectors(model, tok, concepts, vl)
    ref = [chat_prompt(tok, t) for t in ("What is the capital of France?", "Explain photosynthesis briefly.", PARAPHRASES[0])]
    norm = {L: typical_norm(model, tok, ref, L) for L in inj_layers}
    yes_ids = sorted({enc(tok, w)[0] for w in ("Yes", "yes")})
    no_ids = sorted({enc(tok, w)[0] for w in ("No", "no")})
    rng = np.random.RandomState(seed)
    trials = [dict(concept=-1, s=0.0, L=0, para=int(rng.randint(len(PARAPHRASES))), trial=int(rng.randint(1, 900))) for _ in range(n_null)]
    for ci in range(len(concepts)):
        for L in inj_layers:
            for s in strengths:
                trials.append(dict(concept=ci, s=float(s), L=L, para=int(rng.randint(len(PARAPHRASES))), trial=int(rng.randint(1, 900))))
    S, gaps, texts = [], [], []
    for k, t in enumerate(trials):
        prompt = chat_prompt(tok, f"Trial {t['trial']}.\n\n{PARAPHRASES[t['para']]}")
        L = t["L"] or inj_layers[0]
        with Injector(model, L) as inj:
            vec = None if t["concept"] < 0 else vecs[concepts[t["concept"]]] * (t["s"] * norm[L])
            st, gap, txt = one_trial(model, tok, prompt, inj, vec, yes_ids, no_ids, gen_tokens)
        S.append(st), gaps.append(gap), texts.append(txt)
        if k % 100 == 0:
            log(f"[introspect] {k}/{len(trials)}")
    low = [x.lower() for x in texts]
    said_yes = np.array([l.strip().startswith("yes") for l in low])
    named = np.array([(t["concept"] >= 0) and (concepts[t["concept"]] in l) for t, l in zip(trials, low)])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(f.with_name("introspect.tmp.npz"), states=np.stack(S), gap=np.array(gaps), said_yes=said_yes, named=named,
             concept=np.array([t["concept"] for t in trials]), s=np.array([t["s"] for t in trials]),
             L=np.array([t["L"] for t in trials]), para=np.array([t["para"] for t in trials]), norm=np.array([norm[L] for L in inj_layers]),
             inj_layers=np.array(inj_layers))
    f.with_name("introspect.tmp.npz").rename(f)
    with open(out_dir / "introspect_text.jsonl", "w") as fh:
        for t, x in zip(trials, texts):
            fh.write(json.dumps({**t, "concept_word": concepts[t["concept"]] if t["concept"] >= 0 else None, "text": x}) + "\n")
    log(f"[introspect] {len(trials)} trials; said-yes rate on injected = {said_yes[np.array([t['s'] > 0 for t in trials])].mean():.3f}, null = "
        f"{said_yes[np.array([t['s'] == 0 for t in trials])].mean():.3f}")
    return len(trials)
