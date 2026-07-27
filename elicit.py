"""Forecaster-under-audit harness: ask every target once, in a fresh context.

The one design constraint that makes the audit mean anything: a target must be asked
with NO sight of any sibling target in its family. If the model sees its own P(A) and
P(B) before it is asked P(A and B) it will paper over its own incoherence and we
measure nothing. One `claude -p` process per target buys that for free, so targets
from the same family are never batched into one prompt, and the prompt for a target
shows the resolution criteria only for the base events its own expr/given names.

Writes out/forecasts_<model>.json: a list of
    {"tid", "model", "p", "rationale", "asked_utc", "arm", "sample"}
one per target per draw, in question-file order. Unusable answers get p=None + "error".

Three things vary the forecaster, and each gets its own file and its own identity in
the "model" field, because coherence.py and dutchbook.py group books by that field
alone -- two arms sharing one label would be merged into one incoherent book:

    default          "<model>"                out/forecasts_<model>.json
    --tools          "<model>+search"         out/forecasts_<model>_search.json
    --joint          "<model>+joint"          out/forecasts_<model>_joint.json

`arm` is "fresh" or "joint" in every record (scoring.py groups on (model, arm, tid),
baselines.py keys on f"{model}|{arm}").

--repeat N is the test-retest noise floor: N draws of every target, varying only an
appended nonce comment. Sample 0 renders byte-identically to a --repeat 1 prompt, so
every already-paid cache entry stays valid. Writes out/retest_<model>.json and prints
the within-target SD next to the coherence gap it has to be smaller than.

    python3 elicit.py --selftest                  # exactly one real call, toy target
    python3 elicit.py --dry-run                   # what would this bill? no calls
    python3 elicit.py --limit 8                   # cheap smoke run, same out path
    python3 elicit.py --model claude-sonnet-5 --model claude-opus-5
    python3 elicit.py --tools                     # retrieval-augmented arm
    python3 elicit.py --repeat 5                  # noise floor, 5 draws per target
    python3 elicit.py --joint                     # one prompt per family, all targets
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import algebra
import llm

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
TODAY = "2026-07-26"

PROMPT = """Today's date is {today}. You are forecasting an event whose outcome is not \
yet known, and you have no information about anything that happens after today.

QUESTION
{question}

RESOLUTION CRITERIA
{criteria}

INSTRUCTIONS
- Report the probability you actually believe: the number you would be willing to take \
either side of a bet at. Give a calibrated estimate, not a hedge -- do not drift toward \
0.5 and do not round off to a comfortable number.
{extra}- 0 and 1 are not permitted; p must be strictly between 0 and 1.
- Reply with ONLY this JSON object -- no preamble, no code fence, no extra keys:
{{"p": <float strictly between 0 and 1>, "rationale": "<one sentence>"}}"""

COND_LINE = ("- The question tells you to assume something. Treat that assumption as "
             "certainly true, and report the probability of the asked event in that "
             "world.\n")

STRICT = """

FORMAT REMINDER: the previous reply was not usable. Output exactly one JSON object and \
nothing else: two keys, "p" (a JSON number strictly greater than 0 and strictly less \
than 1) and "rationale" (a one-sentence string). No markdown, no code fence, no text \
before or after it."""

# The joint arm: every target of one family in one prompt, so the model can see its own
# numbers side by side. Deliberately says NOTHING about coherence, consistency, or the
# probability axioms -- the arm measures how much coherence sight alone buys, and a nudge
# to "be consistent" would be measuring the nudge instead.
JOINT_PROMPT = """Today's date is {today}. You are forecasting events whose outcomes are \
not yet known, and you have no information about anything that happens after today.

Below are {n} probability questions about the same situation. Answer all of them.

RESOLUTION CRITERIA
{criteria}

QUESTIONS
{questions}

INSTRUCTIONS
- For each question report the probability you actually believe: the number you would be \
willing to take either side of a bet at. Give calibrated estimates, not hedges -- do not \
drift toward 0.5 and do not round off to comfortable numbers.
- A question that tells you to assume something is conditional: treat that assumption as \
certainly true, and report the probability of the asked event in that world.
- 0 and 1 are not permitted; every p must be strictly between 0 and 1.
- Reply with ONLY a JSON array of exactly {n} objects, one per question, in the same order \
as the questions above -- no preamble, no code fence, no extra keys:
[{{"tid": "<the id shown in brackets>", "p": <float strictly between 0 and 1>, \
"rationale": "<one sentence>"}}]"""

STRICT_JOINT = """

FORMAT REMINDER: the previous reply was not usable. Output exactly one JSON array and \
nothing else, with exactly {n} objects in the same order as the questions, each with keys \
"tid" (the id shown in brackets), "p" (a JSON number strictly greater than 0 and strictly \
less than 1) and "rationale" (a one-sentence string). No markdown, no code fence, no text \
before or after it."""

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def event_names(target, base_events):
    """Base events named by this target's expr/given, in order of first appearance.

    Deliberately does NOT return the family's other base events: showing a sibling
    event the target never mentions would be the exact contamination we are avoiding.
    """
    out = []
    for expr in (target.get("expr", ""), target.get("given", "") or ""):
        for tok in _IDENT.findall(expr):
            if tok in base_events and tok not in out:
                out.append(tok)
    return out


def nonce(sample):
    """Draw separator for --repeat.

    Sample 0 MUST return "" so the rendered prompt is byte-identical to a --repeat 1
    prompt: hundreds of paid replies are already cached under those exact bytes.
    Later draws append a comment, which changes the cache key and nothing else.
    """
    return "" if not sample else f"\n<!-- sample {int(sample)} -->"


def _criteria_block(base, names):
    return "\n".join(
        f"- Event: {base[n]['text']}\n"
        f"  Resolution: {base[n]['resolution']}\n"
        f"  Resolution source: {base[n]['source']}\n"
        f"  Must resolve by: {base[n]['resolve_by']}"
        for n in names
    )


def build_prompt(family, target, today=TODAY, sample=0):
    base = family["base_events"]
    names = event_names(target, base)
    if not names:  # cannot happen for a validated family; fail loudly if it does
        raise ValueError(f"{target['tid']}: expr names no base event")
    return PROMPT.format(
        today=today,
        question=target["text"].strip(),
        criteria=_criteria_block(base, names),
        extra=COND_LINE if target.get("kind") == "conditional" else "",
    ) + nonce(sample)


def build_joint_prompt(family, today=TODAY, sample=0):
    """Every target of one family in one prompt, numbered, tids shown in brackets."""
    base = family["base_events"]
    names = []
    for t in family["targets"]:
        for n in event_names(t, base):
            if n not in names:
                names.append(n)
    targets = family["targets"]
    questions = "\n".join(
        f"{i + 1}. [{t['tid']}] {t['text'].strip()}" for i, t in enumerate(targets)
    )
    return JOINT_PROMPT.format(
        today=today, n=len(targets),
        criteria=_criteria_block(base, names),
        questions=questions,
    ) + nonce(sample)


def _p_and_rationale(obj, where=""):
    """(p, one-line rationale) from one reply object. Raises ValueError if unusable."""
    if not isinstance(obj, dict) or "p" not in obj:
        raise ValueError(f"no 'p' key{where} in {str(obj)[:160]}")
    try:
        p = float(obj["p"])
    except (TypeError, ValueError):
        raise ValueError(f"p is not a number{where}: {obj['p']!r}") from None
    if not (p > 0.0 and p < 1.0):
        raise ValueError(f"p={p}{where} outside the open interval (0, 1)")
    rat = obj.get("rationale") or obj.get("reason") or ""
    return p, " ".join(str(rat).split())[:600]


def parse_reply(raw):
    """(p, rationale). Raises ValueError if unusable or p is outside (0, 1)."""
    obj = llm.extract_json(raw)
    if isinstance(obj, list):
        if len(obj) != 1 or not isinstance(obj[0], dict):
            raise ValueError(f"expected one JSON object, got a list of {len(obj)}")
        obj = obj[0]
    return _p_and_rationale(obj)


def parse_joint(raw, targets):
    """[(p, rationale)] aligned to `targets`. Raises ValueError if unusable.

    Matches on the echoed tid when the model returns them all, and falls back to
    position otherwise -- but never silently accepts a short or long array, because a
    length mismatch means we cannot know which answer belongs to which question.
    """
    obj = llm.extract_json(raw)
    if isinstance(obj, dict):
        for k in ("forecasts", "answers", "results", "predictions", "questions"):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
    if not isinstance(obj, list):
        raise ValueError(f"expected a JSON array of {len(targets)} objects, got "
                         f"{type(obj).__name__}")
    if len(obj) != len(targets):
        raise ValueError(f"expected {len(targets)} objects, got {len(obj)}")
    by_tid = {it["tid"].strip(): it for it in obj
              if isinstance(it, dict) and isinstance(it.get("tid"), str)}
    aligned = all(t["tid"] in by_tid for t in targets)
    out = []
    for i, t in enumerate(targets):
        item = by_tid[t["tid"]] if aligned else obj[i]
        out.append(_p_and_rationale(item, where=f" for {t['tid']}"))
    return out


def arm_label(model, tools=False, arm="fresh"):
    """Forecaster identity written into the "model" field of every record.

    coherence.py and dutchbook.py group books by this field ALONE -- they know nothing
    about "arm" or "sample". If the fresh and joint arms both called themselves
    "claude-sonnet-5" those two loaders would merge them into a single book (one keeping
    the first p per tid, the other the last) and both coherence audits -- the headline of
    the whole repo -- would be measuring a book no forecaster ever quoted. So the arm
    goes in the label too. scoring.py/baselines.py additionally split on "arm", which
    makes the label redundant there but never wrong.
    """
    return f"{model}{'+search' if tools else ''}{'+joint' if arm == 'joint' else ''}"


def _blank(tid, model, tools, arm, sample):
    return {"tid": tid, "model": arm_label(model, tools, arm), "p": None, "rationale": "",
            "asked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "arm": arm, "sample": int(sample)}


def ask(family, target, model, tools=False, effort=llm.DEFAULT_EFFORT, retry=True,
        sample=0, verbose=True):
    """One target, one fresh process. Returns a forecast record."""
    tid = target["tid"]
    prompt = build_prompt(family, target, sample=sample)
    rec = _blank(tid, model, tools, "fresh", sample)
    errs = []
    for i, pr in enumerate([prompt] + ([prompt + STRICT] if retry else [])):
        try:
            raw = llm.call(pr, model, tag=tid, tools=tools, effort=effort,
                           verbose=verbose)
            p, rat = parse_reply(raw)
            rec["p"], rec["rationale"] = p, rat
            if i:
                rec["retried"] = True
            return rec
        except llm.FatalCallError:
            raise  # bad model name or broken auth: abort, don't write 200 nulls
        except Exception as e:  # noqa: BLE001 - never let one target kill the run
            errs.append(f"{type(e).__name__}: {e}")
            if verbose:
                print(f"  [{tid}] attempt {i + 1} unusable: {str(e)[:180]}", flush=True)
            # Show what it actually said, read straight from the cache: never re-call.
            cp = llm.cache_path(pr, model, tools=tools, effort=effort)
            if cp.exists():
                rec["raw"] = json.loads(cp.read_text())["result"][:400]
    rec["error"] = " | ".join(errs)
    return rec


def ask_joint(family, model, tools=False, effort=llm.DEFAULT_EFFORT, retry=True,
              sample=0, verbose=True):
    """One family, one process, all its targets at once. Returns one record per target.

    All-or-nothing: a family whose array cannot be aligned yields p=None for every one
    of its targets, so the record count still matches the target count exactly.
    """
    targets = family["targets"]
    prompt = build_joint_prompt(family, sample=sample)
    tag = f"joint:{family['family_id']}"
    recs = [_blank(t["tid"], model, tools, "joint", sample) for t in targets]
    errs = []
    strict = STRICT_JOINT.format(n=len(targets))
    for i, pr in enumerate([prompt] + ([prompt + strict] if retry else [])):
        try:
            raw = llm.call(pr, model, tag=tag, tools=tools, effort=effort,
                           verbose=verbose)
            for rec, (p, rat) in zip(recs, parse_joint(raw, targets)):
                rec["p"], rec["rationale"] = p, rat
                if i:
                    rec["retried"] = True
            return recs
        except llm.FatalCallError:
            raise
        except Exception as e:  # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")
            if verbose:
                print(f"  [{tag}] attempt {i + 1} unusable: {str(e)[:180]}", flush=True)
            cp = llm.cache_path(pr, model, tools=tools, effort=effort)
            for rec in recs:
                rec["p"], rec["rationale"] = None, ""
                if cp.exists():
                    rec["raw"] = json.loads(cp.read_text())["result"][:400]
    for rec in recs:
        rec["error"] = " | ".join(errs)
    return recs


def plan(families, limit=None, joint=False, repeat=1):
    """The units of work, in stable output order.

    A unit is one `claude -p` call: one (target, sample) in the fresh arm, one
    (family, sample) in the joint arm. Sample-major ordering matters -- it puts every
    sample-0 record first, which is the draw coherence.py keeps when it dedupes a
    repeat file by (model, tid), and it makes --repeat 1 byte-for-byte the same output
    order as before --repeat existed.
    """
    pairs = [(f, t) for f in families for t in f["targets"]]
    if limit:
        pairs = pairs[:limit]
    if joint:
        # A family must be asked whole or not at all, so --limit selects the families
        # that its target window touches and asks each of them completely.
        keep, fams = {id(f) for f, _ in pairs}, []
        for f in families:
            if id(f) in keep:
                fams.append(f)
        return [(s, f, None) for s in range(max(1, repeat)) for f in fams]
    return [(s, f, t) for s in range(max(1, repeat)) for f, t in pairs]


def elicit(model, families, workers=8, use_tools=False, effort=llm.DEFAULT_EFFORT,
           limit=None, repeat=1, joint=False, verbose=True):
    """Ask every target of every family, `repeat` times. Order is stable."""
    units = plan(families, limit=limit, joint=joint, repeat=repeat)
    chunks = [None] * len(units)
    n_expected = sum(len(f["targets"]) if t is None else 1 for _, f, t in units)

    def work(i):
        s, f, t = units[i]
        if t is None:
            chunks[i] = ask_joint(f, model, tools=use_tools, effort=effort,
                                  sample=s, verbose=verbose)
        else:
            chunks[i] = [ask(f, t, model, tools=use_tools, effort=effort,
                             sample=s, verbose=verbose)]
        if verbose:
            done = sum(len(c) for c in chunks if c is not None)
            got = [r["p"] for r in chunks[i]]
            shown = ("p=%.3f" % got[0] if len(got) == 1 and got[0] is not None
                     else f"{sum(g is not None for g in got)}/{len(got)} ok")
            label = chunks[i][0]["tid"] if t is not None else f["family_id"]
            print(f"  [{done}/{n_expected}] {label}"
                  f"{'' if not s else f' s{s}'}: {shown}", flush=True)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(work, range(len(units))))
    return [r for c in chunks for r in c]


# ------------------------------------------------------------------- noise floor

def _pctile(sorted_vals, q):
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def retest(recs):
    """({tid: {draws, n, mean, sd}}, {mean_sd, median_sd, p90_sd, ...}).

    sd is the sample standard deviation (n-1): the test-retest noise floor of one
    target. A target with fewer than two usable draws gets sd=None and is excluded
    from the spread statistics rather than counted as zero variance.
    """
    per, order = {}, []
    for r in recs:
        if r.get("p") is None:
            continue
        if r["tid"] not in per:
            per[r["tid"]] = {"draws": [], "n": 0, "mean": None, "sd": None}
            order.append(r["tid"])
        per[r["tid"]]["draws"].append(float(r["p"]))
    for tid in order:
        d = per[tid]["draws"]
        n = len(d)
        m = sum(d) / n
        per[tid]["n"] = n
        per[tid]["mean"] = m
        per[tid]["sd"] = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else None
    sds = sorted(v["sd"] for v in per.values() if v["sd"] is not None)
    top = {"n_targets": len(per), "n_targets_with_sd": len(sds),
           "mean_sd": None, "median_sd": None, "p90_sd": None, "max_sd": None}
    if sds:
        top.update(mean_sd=sum(sds) / len(sds), median_sd=_pctile(sds, 50),
                   p90_sd=_pctile(sds, 90), max_sd=sds[-1])
    return per, top


def axiom_gap(label, path=None):
    """(mean_abs_repair for this forecaster from out/coherence.json, note).

    That is the mean absolute probability the QP projection has to move to make the
    book coherent -- the per-target axiom gap the noise floor has to be smaller than.
    """
    path = pathlib.Path(path or (OUT / "coherence.json"))
    if not path.exists():
        return None, f"{path.name} not written yet -- run coherence.py"
    try:
        agg = (json.loads(path.read_text()).get("aggregates") or {})
    except Exception as e:  # noqa: BLE001
        return None, f"{path.name} unreadable ({e})"
    row = agg.get(label)
    if row is None:
        return None, f"no coherence aggregate for {label!r} (have {sorted(agg)})"
    return row.get("mean_abs_repair"), None


# --------------------------------------------------------------------------- toy

TOY = {
    "family_id": "toy_selftest",
    "domain": "toy",
    "base_events": {
        "A": {
            "text": ("the Bank of Canada's target for the overnight rate is 2.00% or "
                     "lower immediately following its scheduled decision of "
                     "2027-01-27"),
            "resolution": ("Resolves YES if the Bank of Canada's published policy "
                           "interest rate announcement of 2027-01-27 states a target "
                           "for the overnight rate of 2.00% or lower"),
            "source": "Bank of Canada policy interest rate announcements page",
            "resolve_by": "2027-01-28",
        },
        "B": {
            "text": ("the Bank of Canada lowers its target for the overnight rate at "
                     "at least one scheduled decision between 2026-09-01 and "
                     "2027-01-27 inclusive"),
            "resolution": ("Resolves YES if any Bank of Canada scheduled policy "
                           "interest rate announcement in that window sets a target "
                           "for the overnight rate below the target in force "
                           "immediately before it"),
            "source": "Bank of Canada policy interest rate announcements page",
            "resolve_by": "2027-01-28",
        },
    },
    "targets": [
        {"tid": "toy_selftest.A", "kind": "unconditional", "expr": "A",
         "text": ("What is the probability that the Bank of Canada's target for the "
                  "overnight rate is 2.00% or lower immediately following its "
                  "scheduled decision of 2027-01-27?")},
        {"tid": "toy_selftest.B", "kind": "unconditional", "expr": "B",
         "text": ("What is the probability that the Bank of Canada lowers its target "
                  "for the overnight rate in at least one scheduled decision between "
                  "2026-09-01 and 2027-01-27 inclusive?")},
        {"tid": "toy_selftest.AandB", "kind": "unconditional", "expr": "A and B",
         "text": ("What is the probability that BOTH of the following happen: the Bank "
                  "of Canada lowers its target for the overnight rate in at least one "
                  "scheduled decision between 2026-09-01 and 2027-01-27 inclusive, AND "
                  "its target for the overnight rate is 2.00% or lower immediately "
                  "following its scheduled decision of 2027-01-27?")},
        {"tid": "toy_selftest.A_g_B", "kind": "conditional", "expr": "A", "given": "B",
         "text": ("Assume the Bank of Canada lowers its target for the overnight rate "
                  "in at least one scheduled decision between 2026-09-01 and "
                  "2027-01-27 inclusive. Given that, what is the probability that its "
                  "target for the overnight rate is 2.00% or lower immediately "
                  "following its scheduled decision of 2027-01-27?")},
    ],
    "axioms_tested": ["product_rule", "conjunction_bound"],
}


def selftest(model, effort=llm.DEFAULT_EFFORT, tools=False):
    """Exactly ONE real `claude -p` call, on the hand-written toy target."""
    algebra.validate(TOY)  # the toy also has to satisfy the shared contract
    print(f"toy family validates: {len(algebra.atoms(TOY))} atoms, "
          f"{len(TOY['targets'])} targets\n")
    for t in TOY["targets"]:
        print("=" * 78)
        print(f"PROMPT for {t['tid']} ({t['kind']}) -- rendered, not sent")
        print("=" * 78)
        print(build_prompt(TOY, t))
        print()
    print("=" * 78)
    print("JOINT PROMPT for the whole family (--joint) -- rendered, not sent")
    print("=" * 78)
    print(build_joint_prompt(TOY))
    print()
    print("=" * 78)
    print("--repeat nonce check (the ONLY difference between draws)")
    print("=" * 78)
    s0 = build_prompt(TOY, TOY["targets"][0], sample=0)
    s1 = build_prompt(TOY, TOY["targets"][0], sample=1)
    print(f"sample 0 == no-sample render: {s0 == build_prompt(TOY, TOY['targets'][0])}")
    print(f"sample 1 adds {s1[len(s0):]!r} and nothing else: {s1[:len(s0)] == s0}")
    print(f"sample 0 cache key {llm.cache_path(s0, model).stem[:12]} != "
          f"sample 1 {llm.cache_path(s1, model).stem[:12]}\n")
    target = TOY["targets"][-1]  # conditional: exercises 2 criteria blocks + cond line
    print("=" * 78)
    print(f"SENDING exactly one call: {target['tid']}  model={model} effort={effort} "
          f"tools={tools}")
    cached = llm.is_cached(build_prompt(TOY, target), model, tools=tools, effort=effort)
    print(f"already cached: {cached}")
    print("=" * 78)
    t0 = time.time()
    llm.reset_stats()
    rec = ask(TOY, target, model, tools=tools, effort=effort, retry=False)
    print(f"\nrecord: {json.dumps(rec, indent=2)}")
    print(f"\nwall {time.time() - t0:.1f}s  cache hits {llm.STATS['hits']}  "
          f"live calls {llm.STATS['calls']}  cost ${llm.STATS['cost_usd']:.4f}")
    print(llm.cache_summary())
    return rec


# -------------------------------------------------------------------------- main

def _slug(model, tools=False, arm="fresh"):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", arm_label(model, tools, arm)).strip("_")


def out_path(model, tools=False, arm="fresh", override=None):
    """out/forecasts_<model>[_search][_joint].json -- what the audits glob for.

    A smoke run (--limit) writes the same path on purpose: a later full run simply
    overwrites it, and every target it already asked is a free cache hit.
    """
    if override:
        return pathlib.Path(override)
    return OUT / f"forecasts_{_slug(model, tools, arm)}.json"


def retest_path(model, tools=False, arm="fresh", forecasts=None):
    """Sits beside the forecast file it summarises, whatever --out that was."""
    if forecasts is not None:
        p = pathlib.Path(forecasts)
        stem = p.stem[len("forecasts_"):] if p.stem.startswith("forecasts_") else p.stem
        return p.with_name(f"retest_{stem}.json")
    return OUT / f"retest_{_slug(model, tools, arm)}.json"


def _units_to_bill(model, fams, a, arm):
    """(cached, total) prompts this invocation would look up. Makes no calls."""
    units = plan(fams, limit=a.limit, joint=a.joint, repeat=a.repeat)
    n_cached = 0
    for s, f, t in units:
        pr = build_joint_prompt(f, sample=s) if t is None else build_prompt(f, t, sample=s)
        n_cached += bool(llm.is_cached(pr, model, tools=a.tools, effort=a.effort))
    return n_cached, len(units)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", default=None,
                    help="repeatable; default claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tools", action="store_true",
                    help="retrieval arm: allow WebSearch/WebFetch only")
    ap.add_argument("--effort", default=llm.DEFAULT_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--limit", type=int, default=None, help="first N targets only")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="test-retest: N draws per target (sample 0 reuses the cache)")
    ap.add_argument("--joint", action="store_true",
                    help="joint-context arm: one prompt per family, all its targets")
    ap.add_argument("--qdir", default=str(HERE / "questions"))
    ap.add_argument("--out", default=None, help="override the output path (single model)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cache/billing accounting and exit; makes no calls")
    ap.add_argument("--selftest", action="store_true",
                    help="one real call on a toy target, then exit")
    a = ap.parse_args(argv)
    models = a.model or [llm.DEFAULT_MODEL]
    arm = "joint" if a.joint else "fresh"
    if a.repeat < 1:
        ap.error("--repeat must be >= 1")

    if a.selftest:
        selftest(models[0], effort=a.effort, tools=a.tools)
        return 0

    fams, errs = algebra.load_all(a.qdir)
    for e in errs:
        print("  QUESTION-FILE ERROR", e, flush=True)
    if not fams:
        print(f"\nNo valid families in {a.qdir} -- nothing to elicit.\n"
              "questions/*.json may not be written yet (other agents own them). "
              "Re-run once they exist, or use --selftest to check the pipeline.")
        return 1
    n_targets = sum(len(f["targets"]) for f in fams)
    print(f"{len(fams)} families, {n_targets} targets from {a.qdir}"
          + (f" (limited to {a.limit})" if a.limit else ""))
    if a.repeat > 1:
        print(f"repeat {a.repeat}: coherence.py and dutchbook.py dedupe a repeat file by "
              f"(model, tid) with no sample awareness -- coherence keeps sample 0, "
              f"dutchbook keeps sample {a.repeat - 1}.")

    OUT.mkdir(exist_ok=True)
    rc = 0
    for model in models:
        label = arm_label(model, a.tools, arm)
        pre, n_units = _units_to_bill(model, fams, a, arm)
        print(f"\n=== {label} | arm {arm} | {n_units} calls "
              f"({pre} cached, {n_units - pre} to bill) | repeat {a.repeat} | "
              f"workers {a.workers} | effort {a.effort} | "
              f"tools {'WebSearch,WebFetch' if a.tools else 'none'}")
        if a.dry_run:
            print("    --dry-run: no calls made")
            continue
        llm.reset_stats()
        t0 = time.time()
        recs = elicit(model, fams, workers=a.workers, use_tools=a.tools,
                      effort=a.effort, limit=a.limit, repeat=a.repeat, joint=a.joint)
        path = out_path(model, a.tools, arm, a.out if len(models) == 1 else None)
        path.write_text(json.dumps(recs, indent=1))
        bad = [r for r in recs if r["p"] is None]
        ps = [r["p"] for r in recs if r["p"] is not None]
        print(f"--- {label}: wrote {len(recs)} records -> {path}")
        print(f"    cache hits {llm.STATS['hits']}  live calls {llm.STATS['calls']}  "
              f"transport retries {llm.STATS['retries']}  "
              f"parse/range failures {len(bad)}  cost ${llm.STATS['cost_usd']:.3f}  "
              f"wall {time.time() - t0:.1f}s")
        if ps:
            print(f"    p: min {min(ps):.3f} median "
                  f"{sorted(ps)[len(ps) // 2]:.3f} max {max(ps):.3f}")
        for r in bad[:20]:
            print(f"    FAILED {r['tid']}: {r.get('error', '')[:200]}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more failures")

        if a.repeat > 1:
            per, top = retest(recs)
            gap, note = axiom_gap(label)
            rpath = retest_path(model, a.tools, arm,
                                a.out if len(models) == 1 else None)
            rpath.write_text(json.dumps({
                "generated_utc": dt.datetime.now(dt.timezone.utc)
                                   .isoformat(timespec="seconds"),
                "model": label, "arm": arm, "repeat": a.repeat,
                "source_file": str(path),
                **top,
                "axiom_gap_mean_abs_repair": gap,
                "sd_over_gap": (top["mean_sd"] / gap) if (gap and top["mean_sd"]) else None,
                "axiom_gap_note": note,
                "per_tid": per,
            }, indent=1))
            print(f"    retest -> {rpath}")
            x = top["mean_sd"]
            if x is None:
                print("    no target had two usable draws: no noise floor")
            elif gap:
                ratio = x / gap
                print(f"    mean within-target SD = {x:.4f} vs mean axiom gap = "
                      f"{gap:.4f}  (SD/gap = {ratio:.2f})")
                print("    " + ("*** NOISE FLOOR IS NOT BELOW THE GAP: the incoherence "
                                "is not separable from sampling variance ***"
                                if ratio >= 0.7 else
                                "noise floor is well below the gap: the gap survives"))
            else:
                print(f"    mean within-target SD = {x:.4f} vs mean axiom gap = n/a "
                      f"({note})")
            print(f"    median SD {top['median_sd']:.4f}  p90 SD {top['p90_sd']:.4f}  "
                  f"max SD {top['max_sd']:.4f}"
                  if top["mean_sd"] is not None else "")
    return rc


if __name__ == "__main__":
    sys.exit(main())
