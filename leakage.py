"""Hindsight / contamination probes: is this evaluation actually measuring forecasting?

A retrospective forecast evaluation is worthless if the forecaster can tell when the
question was written, or already knows how it turned out. In an earlier blinded
replication of a published multi-agent trading paper, an adversary model recovered 14 of
15 supposedly anonymised company codenames from the news bundles alone -- the blinding
was decoration. The same trick generalises to question sets, so this module runs it
against our own set before anyone is allowed to believe a score.

Three probes, in increasing order of how much they can hurt:

1. ``date_probe``  -- show the model ONLY the target question texts and ask when they were
   written, whether the event has already resolved in its own knowledge, and if so how.
   Any family that comes back ``already_resolved: true`` is a question that must be
   rewritten; we name those loudly. A set that reads as unresolved and dates to roughly
   the true authoring month (2026-07) earns the hindsight-free certificate, which is the
   deliverable here -- everything downstream (coherence, Dutch-book, scoring) is
   conditional on it.

2. ``controls_resolved.json`` -- 18 hand-authored questions about events that resolved
   between 2024-01 and 2025-12, half YES and half NO, matched in apparent difficulty and
   spread over the same sectors as ``questions/``. Every outcome was verified by web
   search; the per-item ``verified`` field records what was checked. These are the
   positive control: a model with the answers memorised is *sharp* on them.

3. ``contamination_gap`` -- elicit probabilities on those controls under the same
   one-question-per-fresh-context protocol used for the real set (each ``llm.call`` is a
   separate process with no shared history, so no control is ever shown beside another),
   then compare Brier and mean confidence |p-0.5| on the resolved past against mean
   confidence on the unresolved future. Near-perfect and confident on the past while
   diffuse on the future is recall, not skill, and the gap is roughly the discount a
   buyer should apply to any backtest-based claim about this model.

Honesty about probe 3, because it is the one most likely to be over-read: *some* of the
gap is real. The past genuinely is more knowable -- a well-calibrated forecaster with no
memorisation at all should still be more confident about 2024 than about 2027, and part
of the gap is that legitimate asymmetry rather than leakage. The gap is therefore an
upper bound on contamination, not a measurement of it. Two further caveats: our controls
are famous events, whereas ``questions/`` is deliberately hard and near-50/50, so
difficulty is not matched across the two pools; and ``written_estimate`` in probe 1
partly measures where the model's training cutoff is, not only what the questions leak.
The clean version of this test is the matched-difficulty pre/post-cutoff split -- the same
question generator, half of it aimed before the model's cutoff and half after -- which
becomes runnable on this repo once ``questions/`` actually resolves (2026-09 .. 2027-06)
and the resolved half can be re-elicited from a later model. Until then: read the gap as
a ceiling, and read probe 1 as the load-bearing result.

Writes ``out/leakage.json``. Self-test: ``python3 leakage.py --selftest`` makes at most
two live model calls (one date probe on a toy family, one control elicitation).
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import sys

import numpy as np

import algebra

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
QDIR = ROOT / "questions"
CONTROLS = ROOT / "controls_resolved.json"

TRUE_WRITTEN = "2026-07"          # the month this question set was authored
NEAR_MONTHS = 2                   # |months off| <= this counts as "dates to now"
FUTURE_FALLBACK_CAP = 20          # self-elicit at most this many future targets
DEFAULT_MODEL = "claude-sonnet-5"

_MONTH = re.compile(r"(\d{4})[-/](\d{1,2})")


# --------------------------------------------------------------------------- llm access

class LLMUnavailable(RuntimeError):
    """llm.py is not importable yet (another module owns it)."""


def _llm():
    """Lazy import so this module never fails at import time if llm.py is absent.

    Every call routes through llm.py's on-disk cache, so a given (model, prompt) is
    billed at most once across all runs and all probes.
    """
    try:
        import llm  # noqa: PLC0415 - deliberately lazy
    except Exception as e:  # noqa: BLE001
        raise LLMUnavailable(f"llm.py not importable ({e}); leakage probes need it") from e
    for fn in ("call", "extract_json"):
        if not hasattr(llm, fn):
            raise LLMUnavailable(f"llm.py has no {fn}(); leakage probes need it")
    return llm


def _call_many(llm, jobs, workers):
    """jobs: [(prompt, model, tag)]. Uses llm.call_many if present, else sequential."""
    if hasattr(llm, "call_many"):
        return llm.call_many(jobs, workers=workers)
    return [llm.call(*j) for j in jobs]


def _parse_json(llm, text, default):
    try:
        obj = llm.extract_json(text)
        return obj if isinstance(obj, dict) else default
    except Exception:  # noqa: BLE001
        return default


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _as_bool(v):
    """JSON-ish truthiness. Models sometimes emit the string "false", and bool("false")
    is True, which would manufacture a hindsight alarm out of nothing."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


# ------------------------------------------------------------------- probe 1: date probe

DATE_PROMPT = """You are auditing a forecasting question set for hindsight contamination.
Below are questions exactly as a forecaster would see them. You have no other context and
no tools: judge from the text and from your own knowledge only.

Answer three things:
1. In what month was this set of questions most likely WRITTEN? (Any clue counts: phrasing,
   named entities, the state of the world the questions presuppose, deadline dates.)
2. Has the event these questions describe ALREADY RESOLVED as of your own knowledge -- do
   you already know whether it happened?
3. If it has already resolved, what was the outcome for the FIRST question listed?

Reply with ONLY this JSON object and nothing else:
{{"written_estimate":"YYYY-MM","already_resolved":true,"outcome":"YES","confidence":0.5,"clues":"one sentence"}}

Field rules: already_resolved is true or false; outcome is "YES", "NO" or "UNKNOWN"
(use "UNKNOWN" whenever already_resolved is false); confidence is 0..1, your confidence in
the already_resolved / outcome judgement; clues is one sentence naming what gave it away.

QUESTIONS:
{qs}"""


def _months_off(estimate, truth=TRUE_WRITTEN):
    """Signed month distance estimate - truth, or None if unparseable."""
    m = _MONTH.search(str(estimate or ""))
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    if not 1 <= mo <= 12:
        return None
    ty, tmo = int(truth[:4]), int(truth[5:7])
    return (y - ty) * 12 + (mo - tmo)


def date_probe(families, model=DEFAULT_MODEL, workers=4):
    """Ask the model to date each family and say whether it already knows the outcome.

    Only the target question *texts* are shown -- no resolution criteria, no source, no
    resolve_by field, no family id -- so the model gets exactly what leaks through the
    forecaster-visible surface. (Deadline dates inside a question's own text are fair
    game: they are visible to any forecaster, and a set whose deadlines give away the
    authoring month is genuinely leaky.)

    One call per family. Returns a dict with per-family rows and the aggregate
    certificate. Never raises for model/parse failures; raises LLMUnavailable only if
    llm.py is missing.
    """
    llm = _llm()
    fams = list(families)
    if not fams:
        return {"n_families": 0, "rows": [], "skipped": "no families"}

    jobs = []
    for f in fams:
        qs = "\n".join(f"{i + 1}. {t['text']}" for i, t in enumerate(f["targets"]))
        jobs.append((DATE_PROMPT.format(qs=qs), model, f"date-{f['family_id']}"))

    raw = _call_many(llm, jobs, workers)
    rows = []
    for f, r in zip(fams, raw):
        g = _parse_json(llm, r, {})
        est = g.get("written_estimate")
        off = _months_off(est)
        try:
            conf = float(g.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        rows.append({
            "family_id": f["family_id"],
            "domain": f.get("domain"),
            "written_estimate": est,
            "months_off": off,
            "already_resolved": _as_bool(g.get("already_resolved")),
            "outcome": g.get("outcome"),
            "confidence": conf,
            "clues": g.get("clues"),
            "parsed": bool(g),
            "raw_head": None if g else str(r)[:200],
        })

    n = len(rows)
    flagged = [r["family_id"] for r in rows if r["already_resolved"]]
    offs = [r["months_off"] for r in rows if r["months_off"] is not None]
    near = [o for o in offs if abs(o) <= NEAR_MONTHS]
    hist = {}
    for r in rows:
        key = str(r["written_estimate"] or "unparsed")
        hist[key] = hist.get(key, 0) + 1
    confs = [r["confidence"] for r in rows if r["confidence"] is not None]

    frac_resolved = len(flagged) / n
    frac_near = (len(near) / len(offs)) if offs else 0.0
    certified = (not flagged) and frac_near >= 0.5 and len(offs) >= max(1, n // 2)

    return {
        "n_families": n,
        "n_parsed": sum(r["parsed"] for r in rows),
        "true_written_month": TRUE_WRITTEN,
        "n_already_resolved": len(flagged),
        "frac_already_resolved": round(frac_resolved, 4),
        "families_claiming_resolved": flagged,
        "written_estimate_hist": dict(sorted(hist.items())),
        "months_off_median": (float(np.median(offs)) if offs else None),
        "months_off_mean_abs": (round(float(np.mean(np.abs(offs))), 2) if offs else None),
        "frac_within_2_months_of_truth": round(frac_near, 4),
        "mean_confidence": (round(float(np.mean(confs)), 3) if confs else None),
        "certified_hindsight_free": bool(certified),
        "certificate_rule": (f"zero families claiming already_resolved AND >=50% of parsed "
                             f"written_estimate within +/-{NEAR_MONTHS} months of {TRUE_WRITTEN}"),
        "caveat": ("written_estimate also measures where the model's training cutoff sits; "
                   "estimates EARLIER than truth are expected and benign, an already_resolved "
                   "hit is not"),
        "rows": rows,
    }


# ------------------------------------------------------- probe 2: the resolved control set

def load_controls(path=CONTROLS):
    """Read and structurally check controls_resolved.json. Returns (controls, errors)."""
    p = pathlib.Path(path)
    if not p.exists():
        return [], [f"{p.name} missing"]
    try:
        data = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return [], [f"{p.name} unreadable ({e})"]
    if not isinstance(data, list):
        return [], [f"{p.name} must be a JSON list"]

    ctrls, errs, seen = [], [], set()
    for i, c in enumerate(data):
        tag = c.get("tid", f"#{i}") if isinstance(c, dict) else f"#{i}"
        if not isinstance(c, dict):
            errs.append(f"{tag}: not an object")
            continue
        for k in ("tid", "text", "resolution", "source", "resolved_on"):
            if not c.get(k):
                errs.append(f"{tag}: missing {k!r}")
        if c.get("outcome") not in (0, 1):
            errs.append(f"{tag}: outcome must be 0 or 1, got {c.get('outcome')!r}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c.get("resolved_on", ""))):
            errs.append(f"{tag}: resolved_on must be YYYY-MM-DD")
        elif not ("2024-01-01" <= c["resolved_on"] <= "2025-12-31"):
            errs.append(f"{tag}: resolved_on {c['resolved_on']} outside 2024-01..2025-12")
        if c.get("tid") in seen:
            errs.append(f"duplicate tid {c.get('tid')!r}")
        seen.add(c.get("tid"))
        ctrls.append(c)
    return ctrls, errs


def controls_summary(ctrls, errs):
    dom = {}
    for c in ctrls:
        d = c.get("domain", "?")
        slot = dom.setdefault(d, {"n": 0, "yes": 0})
        slot["n"] += 1
        slot["yes"] += int(c.get("outcome") == 1)
    return {
        "path": CONTROLS.name,
        "n": len(ctrls),
        "n_yes": sum(int(c.get("outcome") == 1) for c in ctrls),
        "n_no": sum(int(c.get("outcome") == 0) for c in ctrls),
        "by_domain": dict(sorted(dom.items())),
        "window": "2024-01-01 .. 2025-12-31",
        "errors": errs,
    }


# ------------------------------------------------------- probe 3: the contamination gap

ASK_PROMPT = """You are a careful superforecaster. Answer this single question with a
calibrated probability. You have no tools; use your own knowledge and judgement.

QUESTION: {text}

RESOLUTION CRITERION: {resolution}

Reply with ONLY this JSON object and nothing else:
{{"p": 0.5, "rationale": "one sentence"}}
where p is your probability (0..1) that the question resolves YES."""


def _elicit(llm, items, model, workers, tag_prefix):
    """One fresh context per item -- llm.call is a separate process, so no item is ever
    shown alongside another. Returns forecast records per SCHEMA.md."""
    jobs = [(ASK_PROMPT.format(text=it["text"], resolution=it.get("resolution", "")),
             model, f"{tag_prefix}-{it['tid']}") for it in items]
    raw = _call_many(llm, jobs, workers)
    recs = []
    for it, r in zip(items, raw):
        g = _parse_json(llm, r, {})
        try:
            p = float(g.get("p"))
        except (TypeError, ValueError):
            p = None
        if p is not None and not (0.0 <= p <= 1.0):
            p = min(max(p, 0.0), 1.0)
        recs.append({"tid": it["tid"], "model": model, "p": p,
                     "rationale": g.get("rationale"), "asked_utc": _utc()})
    return recs


def _conf(ps):
    return float(np.mean([abs(p - 0.5) for p in ps])) if ps else None


def _future_confidence(model, limit, workers, llm=None):
    """Mean confidence on the unresolved set. Prefers the sealed elicitation from
    elicit.py; self-elicits a capped sample only if that file does not exist yet."""
    fams, errs = algebra.load_all(str(QDIR))
    uncond = {t["tid"] for f in fams for t in f["targets"] if t["kind"] == "unconditional"}
    all_tids = {t["tid"] for f in fams for t in f["targets"]}

    # same slug rule elicit.py uses for out/forecasts_<model>.json. Never the *_search
    # book: that arm had tools, so its confidence is not the comparison we want.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")
    fpath = OUT / f"forecasts_{slug}.json"
    if not fpath.exists():
        cand = [p for p in sorted(OUT.glob(f"forecasts_{slug}*.json"))
                if not p.stem.endswith("_search")]
        if cand:
            fpath = cand[0]

    if fpath.exists():
        try:
            recs = json.loads(fpath.read_text())
        except Exception as e:  # noqa: BLE001
            recs = []
            errs = list(errs) + [f"{fpath.name} unreadable ({e})"]
        ps_all = [float(r["p"]) for r in recs
                  if isinstance(r.get("p"), (int, float)) and r.get("tid") in all_tids]
        ps_unc = [float(r["p"]) for r in recs
                  if isinstance(r.get("p"), (int, float)) and r.get("tid") in uncond]
        return {"source": f"out/{fpath.name}", "n_all": len(ps_all), "n_uncond": len(ps_unc),
                "mean_conf_all": _conf(ps_all), "mean_conf_uncond": _conf(ps_unc),
                "mean_p_uncond": (round(float(np.mean(ps_unc)), 4) if ps_unc else None),
                "notes": errs}

    # fallback: no sealed forecasts yet, sample the future set ourselves.
    # Give the same amount of resolution detail the controls get, or the future questions
    # would look thinner than the past ones and inflate the gap for the wrong reason.
    def _res(f, expr):
        names = [n for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
                 if n in f["base_events"]]
        return " ".join(f["base_events"][n]["resolution"] for n in dict.fromkeys(names))

    items = [{"tid": t["tid"], "text": t["text"], "resolution": _res(f, t["expr"])}
             for f in fams for t in f["targets"] if t["kind"] == "unconditional"]
    cap = limit if limit else FUTURE_FALLBACK_CAP
    items = items[:cap]
    if not items or llm is None:
        return {"source": "unavailable", "n_all": 0, "n_uncond": 0,
                "mean_conf_all": None, "mean_conf_uncond": None, "mean_p_uncond": None,
                "notes": errs + ["no out/forecasts_*.json and nothing to self-elicit"]}
    recs = _elicit(llm, items, model, workers, "future")
    ps = [r["p"] for r in recs if r["p"] is not None]
    return {"source": f"self-elicited fallback ({len(items)} unconditional targets)",
            "n_all": len(ps), "n_uncond": len(ps), "mean_conf_all": _conf(ps),
            "mean_conf_uncond": _conf(ps),
            "mean_p_uncond": (round(float(np.mean(ps)), 4) if ps else None),
            "notes": errs + ["run elicit.py first for the sealed comparison"]}


def contamination_gap(model=DEFAULT_MODEL, limit=None, workers=4, controls=None):
    """Sharpness on the resolved past vs sharpness on the unresolved future.

    Elicits one probability per resolved control (fresh context each, never two controls
    in one prompt), scores Brier and mean confidence |p-0.5| against the known outcomes,
    and subtracts the model's mean confidence on the future question set.

    ``gap = mean_conf(controls) - mean_conf(future)``. A large positive gap alongside a
    near-zero Brier is recall: the model is not forecasting the controls, it is
    remembering them, and any backtest built on that pool overstates skill by roughly
    this margin. Read it as a ceiling on contamination, not a point estimate -- see the
    module docstring for why part of the gap is legitimate.
    """
    llm = _llm()
    ctrls, errs = load_controls() if controls is None else (list(controls), [])
    if limit:
        ctrls = ctrls[:limit]
    if not ctrls:
        return {"skipped": "no usable controls", "errors": errs}

    recs = _elicit(llm, ctrls, model, workers, "ctl")
    rows, ps, ys = [], [], []
    for c, r in zip(ctrls, recs):
        ok = r["p"] is not None
        if ok:
            ps.append(r["p"])
            ys.append(int(c["outcome"]))
        rows.append({"tid": c["tid"], "domain": c.get("domain"), "outcome": c["outcome"],
                     "p": r["p"], "brier": (round((r["p"] - c["outcome"]) ** 2, 4) if ok else None),
                     "confidence": (round(abs(r["p"] - 0.5), 4) if ok else None),
                     "rationale": r["rationale"]})

    if not ps:
        return {"skipped": "no control forecasts parsed", "errors": errs, "rows": rows}

    p = np.array(ps)
    y = np.array(ys, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    conf_ctl = float(np.mean(np.abs(p - 0.5)))
    decided = p != 0.5
    acc = (float(np.mean(((p[decided] > 0.5) == (y[decided] > 0.5)))) if decided.any() else None)
    mean_p_yes = (float(np.mean(p[y == 1])) if (y == 1).any() else None)
    mean_p_no = (float(np.mean(p[y == 0])) if (y == 0).any() else None)

    fut = _future_confidence(model, limit, workers, llm=llm)
    conf_fut = fut.get("mean_conf_uncond")
    if conf_fut is None:
        conf_fut = fut.get("mean_conf_all")
    gap = (conf_ctl - conf_fut) if conf_fut is not None else None

    return {
        "model": model,
        "n_controls_scored": int(len(ps)),
        "past": {
            "brier": round(brier, 4),
            "brier_skill_vs_uninformative": round(1 - brier / 0.25, 4),
            "mean_confidence": round(conf_ctl, 4),
            "accuracy_at_0.5": (round(acc, 4) if acc is not None else None),
            "n_undecided_at_0.5": int((~decided).sum()),
            "mean_p_on_YES_controls": (round(mean_p_yes, 4) if mean_p_yes is not None else None),
            "mean_p_on_NO_controls": (round(mean_p_no, 4) if mean_p_no is not None else None),
            "recall_separation": (round(mean_p_yes - mean_p_no, 4)
                                  if None not in (mean_p_yes, mean_p_no) else None),
        },
        "future": fut,
        "gap_confidence_past_minus_future": (round(gap, 4) if gap is not None else None),
        "implied_backtest_discount": (round(gap / conf_ctl, 4)
                                      if gap is not None and conf_ctl > 1e-9 else None),
        "interpretation": ("fraction of the model's control-set sharpness that does NOT survive "
                           "onto genuinely open questions; an upper bound on contamination, since "
                           "the past is also legitimately more knowable"),
        "clean_test_note": ("matched-difficulty pre/post-cutoff split, runnable once questions/ "
                            "resolves (2026-09..2027-06)"),
        "rows": rows,
        "errors": errs,
    }


# ----------------------------------------------------------------------------- verdict

def _verdict(dp, gap, ctl):
    bits = []
    if dp.get("skipped"):
        bits.append(f"date probe SKIPPED ({dp['skipped']}) -- NO hindsight certificate")
    elif dp.get("n_families"):
        flagged = dp["families_claiming_resolved"]
        if flagged:
            bits.append(f"NOT CERTIFIED: {len(flagged)}/{dp['n_families']} families read as "
                        f"ALREADY RESOLVED and must be rewritten -> {', '.join(flagged)}")
        elif dp["certified_hindsight_free"]:
            bits.append(f"HINDSIGHT-FREE CERTIFIED: 0/{dp['n_families']} families already "
                        f"resolved, {dp['frac_within_2_months_of_truth']:.0%} date within "
                        f"2mo of {TRUE_WRITTEN}")
        else:
            bits.append(f"0/{dp['n_families']} already resolved (good) but only "
                        f"{dp['frac_within_2_months_of_truth']:.0%} date near {TRUE_WRITTEN} "
                        f"(median {dp['months_off_median']} mo) -- no certificate")
    else:
        bits.append("date probe did not run (no families)")

    if gap and not gap.get("skipped"):
        pa = gap["past"]
        g = gap["gap_confidence_past_minus_future"]
        n_yes = sum(int(r["outcome"] == 1) for r in gap["rows"] if r["p"] is not None)
        bits.append(f"resolved controls (n={gap['n_controls_scored']}, {n_yes} YES): "
                    f"Brier {pa['brier']:.3f} (skill {pa['brier_skill_vs_uninformative']:+.2f}), "
                    f"mean conf {pa['mean_confidence']:.3f}")
        if g is None:
            bits.append("future-set confidence unavailable -> gap not computable")
        else:
            cf = gap["future"].get("mean_conf_uncond")
            cf = gap["future"].get("mean_conf_all") if cf is None else cf
            bits.append(f"future conf {cf:.3f} -> contamination gap {g:+.3f} "
                        f"({gap['implied_backtest_discount']:.0%} of past sharpness does not "
                        f"transfer; ceiling, not point estimate)")
    else:
        bits.append("contamination gap did not run")
    return " | ".join(bits)


# --------------------------------------------------------------------------------- run

def run(model=DEFAULT_MODEL, limit=None, workers=4):
    """All three probes -> out/leakage.json (+ compact stdout summary)."""
    OUT.mkdir(exist_ok=True)
    fams, ferrs = algebra.load_all(str(QDIR))
    if limit:
        fams = fams[:limit]
    ctrls, cerrs = load_controls()
    csum = controls_summary(ctrls, cerrs)

    try:
        dp = date_probe(fams, model=model, workers=workers)
    except LLMUnavailable as e:
        dp = {"n_families": len(fams), "skipped": str(e)}
    try:
        gap = contamination_gap(model=model, limit=limit, workers=workers, controls=ctrls or None)
    except LLMUnavailable as e:
        gap = {"skipped": str(e)}

    verdict = _verdict(dp, gap, csum)
    doc = {
        "meta": {"model": model, "limit": limit, "generated_utc": _utc(),
                 "n_families": len(fams), "question_errors": ferrs,
                 "true_written_month": TRUE_WRITTEN},
        "date_probe": dp,
        "controls": csum,
        "contamination_gap": gap,
        "verdict": verdict,
    }
    (OUT / "leakage.json").write_text(json.dumps(doc, indent=1))

    print(f"leakage probes  model={model}  families={len(fams)}  controls={csum['n']} "
          f"({csum['n_yes']} YES / {csum['n_no']} NO)")
    for e in cerrs:
        print("  CONTROL ERR", e)
    if dp.get("skipped"):
        print(f"  date_probe SKIPPED: {dp['skipped']}")
    elif dp.get("n_families"):
        print(f"  date_probe: already_resolved {dp['n_already_resolved']}/{dp['n_families']}"
              f"  written_estimate median offset {dp['months_off_median']} mo vs {TRUE_WRITTEN}"
              f"  near-now {dp['frac_within_2_months_of_truth']:.0%}")
        for k, v in dp["written_estimate_hist"].items():
            print(f"      {k}: {v}")
        for r in dp["rows"]:
            if r["already_resolved"]:
                print(f"    !! REWRITE {r['family_id']}: claims resolved "
                      f"outcome={r['outcome']} conf={r['confidence']} -- {r['clues']}")
    if gap.get("skipped"):
        print(f"  contamination_gap SKIPPED: {gap['skipped']}")
    else:
        pa = gap["past"]
        print(f"  past   n={gap['n_controls_scored']} brier={pa['brier']:.4f} "
              f"skill={pa['brier_skill_vs_uninformative']:+.3f} conf={pa['mean_confidence']:.4f} "
              f"acc={pa['accuracy_at_0.5']} sep={pa['recall_separation']}")
        print(f"  future src={gap['future']['source']} n={gap['future']['n_uncond']} conf="
              f"{gap['future'].get('mean_conf_uncond')}")
        print(f"  GAP    {gap['gap_confidence_past_minus_future']}  "
              f"discount={gap['implied_backtest_discount']}")
    print(f"  verdict: {verdict}")
    return doc


# ---------------------------------------------------------------------------- selftest

TOY = {
    "family_id": "toy_boc_rate",
    "domain": "macro",
    "base_events": {
        "A": {"text": "the Bank of Canada's overnight policy rate is at or below 1.75% "
                      "immediately after its March 2027 decision",
              "resolution": "Resolves YES if the rate published on bankofcanada.ca after the "
                            "March 2027 decision is <= 1.75%",
              "source": "Bank of Canada policy interest rate page", "resolve_by": "2027-03-31"},
        "B": {"text": "the Bank of Canada cuts its overnight policy rate at least once between "
                      "2026-09-01 and 2027-03-31",
              "resolution": "Resolves YES if any decision in the window lowers the target rate",
              "source": "Bank of Canada policy interest rate page", "resolve_by": "2027-03-31"},
    },
    "constraints": [{"type": "implies", "a": "A", "b": "B"}],
    "targets": [
        {"tid": "toy_boc_rate.A", "kind": "unconditional",
         "text": "What is the probability that the Bank of Canada's overnight policy rate is at "
                 "or below 1.75% immediately after its March 2027 decision?", "expr": "A"},
        {"tid": "toy_boc_rate.B", "kind": "unconditional",
         "text": "What is the probability that the Bank of Canada cuts its overnight policy rate "
                 "at least once between 2026-09-01 and 2027-03-31?", "expr": "B"},
        {"tid": "toy_boc_rate.A_g_B", "kind": "conditional", "given": "B",
         "text": "Assume the Bank of Canada cuts its overnight policy rate at least once between "
                 "2026-09-01 and 2027-03-31. What is the probability that its rate is at or "
                 "below 1.75% immediately after the March 2027 decision?", "expr": "A"},
    ],
    "axioms_tested": ["monotonicity", "conditional_consistency"],
}


def selftest(model=DEFAULT_MODEL):
    """At most TWO live calls: one date probe on the toy family, one control elicitation.

    Everything else here is free: schema checks on the toy family and on the 18 controls.
    """
    print("== structural checks (free) ==")
    A = algebra.validate(TOY)
    print(f"toy family {TOY['family_id']}: {len(A)} atoms, {len(TOY['targets'])} targets, "
          f"implies-constraint honoured ({sum(a['A'] and not a['B'] for a in A)} violating atoms)")

    ctrls, errs = load_controls()
    csum = controls_summary(ctrls, errs)
    print(f"controls: n={csum['n']} yes={csum['n_yes']} no={csum['n_no']} "
          f"by_domain={csum['by_domain']}")
    for e in errs:
        print("  ERR", e)
    if errs:
        print("SELFTEST FAILED: control file invalid")
        return 1
    if not ctrls:
        print("SELFTEST FAILED: no controls")
        return 1

    try:
        llm = _llm()
    except LLMUnavailable as e:
        print(f"\n!! SELFTEST INCOMPLETE: {e}")
        print("   structural half passed; re-run once llm.py lands (2 calls, both cached).")
        return 0

    print("\n== live call 1/2: date_probe on the toy family ==")
    dp = date_probe([TOY], model=model, workers=1)
    r = dp["rows"][0]
    print(f"  written_estimate={r['written_estimate']} (offset {r['months_off']} mo vs "
          f"{TRUE_WRITTEN})  already_resolved={r['already_resolved']}  outcome={r['outcome']}  "
          f"conf={r['confidence']}")
    print(f"  clues: {r['clues']}")
    print(f"  certified_hindsight_free={dp['certified_hindsight_free']} "
          f"(n=1, so this is a smoke check, not the real certificate)")
    if not r["parsed"]:
        print(f"  RAW: {r['raw_head']}")

    print("\n== live call 2/2: one control elicitation ==")
    c = ctrls[0]
    rec = _elicit(llm, [c], model, 1, "selftest")[0]
    truth = c["outcome"]
    print(f"  {c['tid']} ({c.get('domain')}) truth={truth} resolved_on={c['resolved_on']}")
    print(f"  p={rec['p']}  brier={None if rec['p'] is None else round((rec['p'] - truth) ** 2, 4)}"
          f"  conf={None if rec['p'] is None else round(abs(rec['p'] - 0.5), 4)}")
    print(f"  rationale: {rec['rationale']}")
    if rec["p"] is None:
        print("  SELFTEST FAILED: control probability did not parse")
        return 1

    print("\nselftest ok (2 live calls, both now cached; full probe set left to the orchestrator)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="hindsight / contamination probes")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap families and controls (cheap smoke run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selftest", action="store_true",
                    help="2 live calls max: one date probe on a toy family, one control")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args.model))
    run(model=args.model, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
