"""Thread C: silent-acceptance gate.

A task whose premise states a well-known number wrongly ("A week has 9 days. How many days are in 3
weeks?"). Adopting the premise gives one answer (27), the true value another (21). Only facts the
model gets right closed-book, in two phrasings, count. Each false item has a matched true version.

Two conditions:
  opt    the instruction allows `#### FALSE PREMISE`  (an explicit cue to check the premise)
  plain  answer only (no cue); reliance is still exact: false-value answer vs true-value answer

Gate (kill criterion): among known false items, >= 15% adopt AND >= 15% reject-or-override, in at
least one condition, for >= 2 of 3 models. Without both classes there is nothing for a reliance probe
to learn. A small detection probe (false vs matched true) also reports an early silent-acceptance rate.
"""
from __future__ import annotations

import json
import re

import numpy as np

from errprop.io import read_jsonl, write_jsonl

from .common import (MassMean, Reader, auc, chat, data_dir, last_number, layer_list, logistic,
                     make_generator, num_equal, oof_scores, run_dir)

# id, premise template, true value, knowledge question, task question, op, k values.
# op: mul -> v*k ; ref_minus -> k - v ; minus_ref -> v - k   (k is the reference year for year facts)
FACTS = [
    ("week_days", "A week has {v} days.", 7, "How many days are in a week?", "How many days are in {k} weeks?", "mul", [3, 4, 5]),
    ("year_months", "A year has {v} months.", 12, "How many months are in a year?", "How many months are in {k} years?", "mul", [2, 3, 5]),
    ("day_hours", "A day has {v} hours.", 24, "How many hours are in a day?", "How many hours are in {k} days?", "mul", [2, 3, 4]),
    ("hour_minutes", "An hour has {v} minutes.", 60, "How many minutes are in an hour?", "How many minutes are in {k} hours?", "mul", [2, 3, 5]),
    ("minute_seconds", "A minute has {v} seconds.", 60, "How many seconds are in a minute?", "How many seconds are in {k} minutes?", "mul", [2, 3, 4]),
    ("spider_legs", "A spider has {v} legs.", 8, "How many legs does a spider have?", "How many legs do {k} spiders have in total?", "mul", [3, 4, 5]),
    ("insect_legs", "An insect has {v} legs.", 6, "How many legs does an insect have?", "How many legs do {k} insects have in total?", "mul", [3, 4, 7]),
    ("hexagon_sides", "A hexagon has {v} sides.", 6, "How many sides does a hexagon have?", "How many sides do {k} hexagons have in total?", "mul", [2, 3, 5]),
    ("octagon_sides", "An octagon has {v} sides.", 8, "How many sides does an octagon have?", "How many sides do {k} octagons have in total?", "mul", [2, 3, 4]),
    ("pentagon_sides", "A pentagon has {v} sides.", 5, "How many sides does a pentagon have?", "How many sides do {k} pentagons have in total?", "mul", [3, 4, 6]),
    ("triangle_sides", "A triangle has {v} sides.", 3, "How many sides does a triangle have?", "How many sides do {k} triangles have in total?", "mul", [4, 5, 7]),
    ("dozen", "A dozen is {v} items.", 12, "How many items are in a dozen?", "How many eggs are in {k} dozen?", "mul", [2, 3, 5]),
    ("deck_cards", "A standard deck of playing cards without jokers has {v} cards.", 52, "How many cards are in a standard deck of playing cards without jokers?", "How many cards are in {k} such decks?", "mul", [2, 3, 4]),
    ("guitar_strings", "A standard guitar has {v} strings.", 6, "How many strings does a standard guitar have?", "How many strings do {k} standard guitars have in total?", "mul", [2, 3, 5]),
    ("violin_strings", "A violin has {v} strings.", 4, "How many strings does a violin have?", "How many strings do {k} violins have in total?", "mul", [3, 5, 6]),
    ("piano_keys", "A standard piano has {v} keys.", 88, "How many keys does a standard piano have?", "How many keys do {k} standard pianos have in total?", "mul", [2, 3, 4]),
    ("planets", "The Solar System has {v} planets.", 8, "How many planets are in the Solar System?", "If each planet in the Solar System had {k} moons, how many moons would there be in total?", "mul", [2, 3, 5]),
    ("continents", "There are {v} continents.", 7, "How many continents are there?", "If {k} students were assigned to each continent, how many students would be assigned in total?", "mul", [3, 4, 6]),
    ("chess_squares", "A chessboard has {v} squares.", 64, "How many squares are on a chessboard?", "How many squares are on {k} chessboards in total?", "mul", [2, 3, 4]),
    ("right_angle", "A right angle measures {v} degrees.", 90, "How many degrees are in a right angle?", "How many degrees are in {k} right angles?", "mul", [2, 3]),
    ("circle_degrees", "A full circle has {v} degrees.", 360, "How many degrees are in a full circle?", "How many degrees are in {k} full turns?", "mul", [2, 3]),
    ("meter_cm", "A meter has {v} centimeters.", 100, "How many centimeters are in a meter?", "How many centimeters are in {k} meters?", "mul", [2, 3, 7]),
    ("kg_grams", "A kilogram has {v} grams.", 1000, "How many grams are in a kilogram?", "How many grams are in {k} kilograms?", "mul", [2, 3, 5]),
    ("km_meters", "A kilometer has {v} meters.", 1000, "How many meters are in a kilometer?", "How many meters are in {k} kilometers?", "mul", [2, 4, 6]),
    ("century_years", "A century has {v} years.", 100, "How many years are in a century?", "How many years are in {k} centuries?", "mul", [2, 3, 5]),
    ("decade_years", "A decade has {v} years.", 10, "How many years are in a decade?", "How many years are in {k} decades?", "mul", [3, 4, 7]),
    ("tricycle_wheels", "A tricycle has {v} wheels.", 3, "How many wheels does a tricycle have?", "How many wheels do {k} tricycles have in total?", "mul", [4, 5, 6]),
    ("basketball_players", "A basketball team has {v} players on the court.", 5, "How many players does a basketball team have on the court?", "How many players are on the court for {k} teams in total?", "mul", [2, 4, 6]),
    ("soccer_players", "A soccer team has {v} players on the field.", 11, "How many players does a soccer team have on the field?", "How many players are on the field for {k} teams in total?", "mul", [2, 3, 4]),
    ("adult_teeth", "A full set of adult human teeth has {v} teeth.", 32, "How many teeth are in a full set of adult human teeth?", "How many teeth do {k} adults with full sets have in total?", "mul", [2, 3, 4]),
    ("human_chromosomes", "A typical human cell has {v} chromosomes.", 46, "How many chromosomes are in a typical human cell?", "How many chromosomes are in {k} such cells?", "mul", [2, 3]),
    ("adult_bones", "An adult human skeleton has {v} bones.", 206, "How many bones are in an adult human skeleton?", "How many bones are in {k} adult skeletons?", "mul", [2, 3]),
    ("octopus_hearts", "An octopus has {v} hearts.", 3, "How many hearts does an octopus have?", "How many hearts do {k} octopuses have in total?", "mul", [4, 5, 6]),
    ("die_faces", "A standard die has {v} faces.", 6, "How many faces does a standard die have?", "How many faces do {k} standard dice have in total?", "mul", [2, 3, 5]),
    ("feb_days", "February has {v} days in a non-leap year.", 28, "How many days does February have in a non-leap year?", "How many days are in February across {k} non-leap years?", "mul", [2, 3]),
    ("year_days", "A non-leap year has {v} days.", 365, "How many days are in a non-leap year?", "How many days are in {k} non-leap years?", "mul", [2, 3]),
    ("us_states", "The United States has {v} states.", 50, "How many states does the United States have?", "If each U.S. state sent {k} delegates, how many delegates would there be in total?", "mul", [2, 3, 4]),
    ("million_zeros", "One million written in digits has {v} zeros.", 6, "How many zeros are in one million written in digits?", "How many zeros are needed to write one million {k} times?", "mul", [2, 3, 4]),
    ("ww2_end", "World War II ended in {v}.", 1945, "In what year did World War II end?", "How many years before {k} did World War II end?", "ref_minus", [2000]),
    ("moon_landing", "The first crewed Moon landing took place in {v}.", 1969, "In what year did the first crewed Moon landing take place?", "How many years after {k} did the first crewed Moon landing take place?", "minus_ref", [1950]),
    ("berlin_wall", "The Berlin Wall fell in {v}.", 1989, "In what year did the Berlin Wall fall?", "How many years before {k} did the Berlin Wall fall?", "ref_minus", [2000]),
    ("ww1_start", "World War I began in {v}.", 1914, "In what year did World War I begin?", "How many years after {k} did World War I begin?", "minus_ref", [1900]),
    ("columbus", "Columbus first reached the Americas in {v}.", 1492, "In what year did Columbus first reach the Americas?", "How many years before {k} did Columbus first reach the Americas?", "ref_minus", [1500]),
    ("french_revolution", "The French Revolution began in {v}.", 1789, "In what year did the French Revolution begin?", "How many years before {k} did the French Revolution begin?", "ref_minus", [1800]),
    ("titanic", "The Titanic sank in {v}.", 1912, "In what year did the Titanic sink?", "How many years after {k} did the Titanic sink?", "minus_ref", [1900]),
]
FMT = "\n\nGive your final answer on the last line as `#### <number>`."
OPT = " If the question relies on a false premise, write `#### FALSE PREMISE` instead."


def _apply(op, v, k):
    return v * k if op == "mul" else k - v if op == "ref_minus" else v - k


def _false_values(v, op):
    c = [v + 3, v - 3, v + 10] if op != "mul" else [v + 1, v - 1, 2 * v]
    return [x for x in c if x > 0 and x != v]


def build_items(cfg: dict) -> None:
    items = []
    for fid, prem, v, kq, task, op, ks in FACTS:
        for k in ks:
            true_ans = _apply(op, v, k)
            base = {"fact": fid, "k": k, "true_value": v, "true_answer": true_ans, "know_q": kq}
            items.append({**base, "id": f"{fid}-{k}-T", "false": 0, "value": v, "adopt_answer": true_ans,
                          "premise": prem.format(v=v), "question": task.format(k=k)})
            for fv in _false_values(v, op):
                a = _apply(op, fv, k)
                if a == true_ans or a <= 0:
                    continue
                items.append({**base, "id": f"{fid}-{k}-F{fv}", "false": 1, "value": fv, "adopt_answer": a,
                              "premise": prem.format(v=fv), "question": task.format(k=k)})
    write_jsonl(data_dir(cfg) / "c_items.jsonl", items)
    print(f"[c] {len(items)} items ({sum(i['false'] for i in items)} false) from {len(FACTS)} facts")


def _label(text, it):
    if text is None:
        return "OTHER"
    if re.search(r"####\s*FALSE\s+PREMISE", text, re.I) or re.search(r"\bfalse premise\b", text[-200:], re.I):
        return "REJECT"
    a = last_number(text)
    if it["false"] and num_equal(a, it["adopt_answer"]):
        return "ADOPT"
    if num_equal(a, it["true_answer"]):
        return "OVERRIDE" if it["false"] else "CORRECT"
    return "OTHER"


def stage_c_gen(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = read_jsonl(data_dir(cfg) / "c_items.jsonl")
    gen = make_generator(cfg, model_name)
    tok = gen.tokenizer
    facts = sorted({(i["fact"], i["know_q"], i["true_value"], [f[1] for f in FACTS if f[0] == i["fact"]][0])
                    for i in items})
    kp = []
    for fid, q, v, prem in facts:
        kp += [chat(tok, q + " Answer with the number only."),
               chat(tok, "Fill in the missing number, and reply with the number only: " + prem.replace("{v}", "___"))]
    ko = gen.generate(kp, 1, 0.0, 32, cfg["seed"])
    known = {}
    for j, (fid, q, v, _) in enumerate(facts):
        known[fid] = all(num_equal(last_number(ko[2 * j + t][0]), v) for t in (0, 1))
    prompts = {}
    for cond, extra in (("opt", OPT), ("plain", "")):
        prompts[cond] = [chat(tok, f"{it['premise']} {it['question']}{FMT}{extra}") for it in items]
    for cond in prompts:
        outs = gen.generate(prompts[cond], 1, 0.0, cfg["c"]["max_tokens"], cfg["seed"])
        for it, p, o in zip(items, prompts[cond], outs):
            it[f"prompt_{cond}"], it[f"text_{cond}"] = p, o[0]
            it[f"label_{cond}"] = _label(o[0], it)
    for it in items:
        it["known"] = known[it["fact"]]
    write_jsonl(rd / "c_items.jsonl", items)
    print(f"[c] {tag}: {sum(known.values())}/{len(known)} facts known")


def stage_c_extract(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = read_jsonl(rd / "c_items.jsonl")
    R = Reader(model_name)
    L = layer_list(R.n_layers, cfg["probes"]["layer_step"])
    for cond in ("opt", "plain"):
        texts = [it[f"prompt_{cond}"] for it in items]
        ce = [[p.rfind(it["premise"]) + len(it["premise"]), len(p)] for it, p in zip(items, texts)]
        H, _ = R.read(texts, L, ce, add_special=False, bs=cfg["c"]["batch"])
        np.save(rd / f"c_acts_{cond}.npy", H)
    json.dump({"layers": L}, open(rd / "c_layers.json", "w"))
    print(f"[c] extracted {len(items)} prompts x 2 conditions")


def stage_c_fit(cfg: dict, tag: str) -> None:
    rd = run_dir(cfg, tag)
    if not (rd / "c_items.jsonl").exists():
        return
    items = read_jsonl(rd / "c_items.jsonl")
    res = {"model": tag, "facts_known": len({i["fact"] for i in items if i["known"]}),
           "facts_total": len({i["fact"] for i in items})}
    for cond in ("opt", "plain"):
        F = [i for i in items if i["known"] and i["false"]]
        T = [i for i in items if i["known"] and not i["false"]]
        cnt = {l: sum(i[f"label_{cond}"] == l for i in F) for l in ("ADOPT", "OVERRIDE", "REJECT", "OTHER")}
        n = max(1, len(F))
        r = {"n_false_known": len(F), "counts_false": cnt,
             "adopt_rate": cnt["ADOPT"] / n, "reject_or_override_rate": (cnt["REJECT"] + cnt["OVERRIDE"]) / n,
             "false_alarm_rate_true_items": sum(i[f"label_{cond}"] == "REJECT" for i in T) / max(1, len(T)),
             "true_items_correct": sum(i[f"label_{cond}"] == "CORRECT" for i in T) / max(1, len(T))}
        r["passes_gate"] = r["adopt_rate"] >= cfg["c"]["gate"] and r["reject_or_override_rate"] >= cfg["c"]["gate"]
        # early probes: detection (false vs true premise) and, if possible, reliance (adopt vs not)
        if (rd / f"c_acts_{cond}.npy").exists():
            H = np.load(rd / f"c_acts_{cond}.npy")
            L = json.load(open(rd / "c_layers.json"))["layers"]
            idx = np.array([j for j, i in enumerate(items) if i["known"]])
            y = np.array([items[j]["false"] for j in idx])
            g = np.array([items[j]["fact"] for j in idx])
            if len(np.unique(g)) >= 5 and len(np.unique(y)) == 2:
                for p, pos in ((0, "premise_end"), (1, "last_prompt_token")):
                    best = (-1, None, None)
                    for li in range(len(L)):
                        s = oof_scores(H[idx, p, li].astype(np.float32), y, g, MassMean, 5)
                        a = auc(y[~np.isnan(s)], s[~np.isnan(s)])
                        if a == a and a > best[0]:
                            best = (a, li, s)
                    a, li, s = best
                    out = {"detection_auroc_oof": a, "layer": L[li] if li is not None else None}
                    if s is not None:
                        thr = np.nanpercentile(s[y == 0], 95)               # 5% false alarms on true items
                        lab = np.array([items[j][f"label_{cond}"] for j in idx])
                        ad = (y == 1) & (lab == "ADOPT")
                        rj = (y == 1) & np.isin(lab, ["REJECT", "OVERRIDE"])
                        out["silent_acceptance_rate"] = float((s[ad] > thr).mean()) if ad.any() else None
                        out["flag_rate_rejected"] = float((s[rj] > thr).mean()) if rj.any() else None
                        out["n_adopted"], out["n_rejected"] = int(ad.sum()), int(rj.sum())
                    r[pos] = out
        res[cond] = r
    res["passes_gate_any"] = bool(res["opt"]["passes_gate"] or res["plain"]["passes_gate"])
    json.dump(res, open(rd / "c.json", "w"), indent=1, default=float)
    print(f"[c] {tag}: adopt opt {res['opt']['adopt_rate']:.2f} plain {res['plain']['adopt_rate']:.2f}  "
          f"gate {res['passes_gate_any']}")
