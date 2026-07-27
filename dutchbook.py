"""Dutch-book auditor: turn quoted probabilities into an explicit arbitrage, then price it.

de Finetti's operational reading of a probability judgment: quoting p for event E means
you will take either side of a bet on E at price p. `algebra.payoff_matrix(family, prices)`
gives P[i][j], the payoff on atom i of a **1-unit bet on target j at the forecaster's own
quoted price** (conditional bets pay 0 -- are called off -- outside their conditioning
event). A stake vector s says what we do: s_j > 0 = we BUY bet j from them, s_j < 0 = we
SELL it to them, |s_j| = size in units.

The audit is one linear program. With m atoms, n targets, variables x = [s_1..s_n, g]:

    maximize    g
    subject to  sum_j s_j * P[i][j] >= g        for every atom i = 1..m
                -stake_cap <= s_j <= stake_cap  for every target j
                g free

i.e. maximize the *worst-case-over-all-atoms* profit of a bounded book built entirely at
the forecaster's own prices. Fed to scipy.optimize.linprog (HiGHS) in its minimize form:

    c     = [0, ..., 0, -1]                      (minimize -g)
    A_ub  = [ -P | 1 ]   (m x (n+1))             (from  -P s + g <= 0)
    b_ub  = [0, ..., 0]
    bounds= [(-cap, cap)] * n + [(None, None)]

s = 0 is always feasible with worst-case profit 0, so the optimal value g* >= 0 always.
By de Finetti / LP duality, g* > 0 **iff** no distribution over atoms reproduces the quoted
numbers, i.e. iff the judgments are incoherent: the dual assigns atom weights lambda >= 0,
sum lambda = 1, and g* = min_lambda sum_j |(P^T lambda)_j| * cap, which is 0 exactly when
some lambda prices every bet at its quote. That makes this module an INDEPENDENT check on
coherence.py -- it shares only algebra.py, and deliberately does not import coherence.

Headline number: g / total_abs_stake, "you lose $X per $100 staked no matter what happens".

Public API
----------
    dutch_book(family, prices, stake_cap=1.0, tol=1e-9) -> dict
    dollars_per_100(result) -> float
    run(qdir="questions", forecasts_glob="out/forecasts_*.json") -> dict   (writes out/dutchbook.json)

`python3 dutchbook.py` runs a hand-checkable self-test, then run() if forecasts exist.
"""
from __future__ import annotations

import glob as _glob
import json
import math
import pathlib
import re
import statistics
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import linprog

import algebra

__all__ = ["dutch_book", "dollars_per_100", "run"]

OUT_PATH = "out/dutchbook.json"


# --------------------------------------------------------------------------- #
# legibility helpers: turn a target into something a non-specialist can read
# --------------------------------------------------------------------------- #
def _short(txt, limit=70):
    """Collapse whitespace and clip to `limit` chars at a word boundary."""
    txt = " ".join(str(txt).split())
    if len(txt) <= limit:
        return txt
    head = txt[:limit].rsplit(" ", 1)[0]
    return (head if len(head) >= limit // 2 else txt[:limit]) + "..."


def _expr_label(expr, base_events, limit=70):
    """Rewrite a boolean formula with base-event *texts* substituted for their names."""

    def repl(m):
        tok = m.group(0)
        if tok in ("and", "or", "not"):
            return tok.upper()  # reads better next to long event sentences
        ev = base_events.get(tok)
        if isinstance(ev, dict) and ev.get("text"):
            return _short(ev["text"], limit)
        return tok

    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, expr)


def _target_label(family, target, limit=70):
    """'P(<event text>)' or 'P(<event text> | <condition text>)'."""
    be = family.get("base_events", {})
    lab = f"P({_expr_label(target['expr'], be, limit)})"
    if target.get("kind") == "conditional" and target.get("given"):
        lab = f"P({_expr_label(target['expr'], be, limit)} | {_expr_label(target['given'], be, limit)})"
    return lab


def _atom_label(atom):
    return " ".join(f"{k}={'T' if atom[k] else 'F'}" for k in sorted(atom))


def _fmt_price(p):
    s = f"{p:.2f}"
    if abs(float(s) - p) > 5e-4:
        s = f"{p:.4f}".rstrip("0")
    return s


def _fmt_stake(x):
    s = f"{abs(x):.2f}"
    if float(s) == 0.0:
        s = f"{abs(x):.4f}"
    return s


# --------------------------------------------------------------------------- #
# the auditor
# --------------------------------------------------------------------------- #
def dutch_book(family, prices, stake_cap=1.0, tol=1e-9):
    """Maximize the guaranteed (worst-case-over-atoms) profit of a book at their prices.

    Parameters
    ----------
    family : dict      a family per SCHEMA.md (only base_events/targets/constraints are used)
    prices : sequence  quoted probabilities, aligned with family["targets"]
    stake_cap : float  per-bet position limit; stakes live in [-cap, +cap]
    tol : float        arbitrage / negligible-stake threshold

    Returns
    -------
    dict with keys
        g                  guaranteed profit of the returned book, per unit stake cap
                           (= min over atoms of the returned stakes' payoff; >= 0 always)
        stakes             list aligned with targets; + = we buy at their price, - = we sell
        per_atom_profit    list aligned with algebra.atoms(family)
        total_abs_stake    sum |s_j|
        g_per_unit_stake   g / total_abs_stake (0.0 if no stake)
        arbitrage          bool, g > tol
        status             solver status string
        certificate        human-readable lines: one per non-negligible stake, then one
                           net line stating the guaranteed profit and that it holds on all atoms
        plus: family_id, n_atoms, n_targets, tids, prices, atom_labels, target_labels,
              stake_cap, tol, lp_g (raw LP objective), prices_out_of_range, success

    Note on ties: the LP can have many optimizers (payoff matrices are often singular).
    When the optimal value is <= tol there is no arbitrage, and the auditor reports the
    honest recommended book -- the empty one (all stakes 0, g = 0) -- rather than an
    arbitrary zero-profit vertex. When there IS arbitrage the LP's optimizer is returned
    as-is; `g` is always recomputed from the returned stakes through algebra.payoff_matrix,
    so the certificate is never stronger than the book it prints.
    """
    targets = family["targets"]
    n = len(targets)
    prices = [float(p) for p in prices]
    if len(prices) != n:
        raise ValueError(f"{family.get('family_id')}: {len(prices)} prices for {n} targets")
    if not all(math.isfinite(p) for p in prices):
        raise ValueError(f"{family.get('family_id')}: non-finite price in {prices}")
    cap = float(stake_cap)
    if not (cap > 0 and math.isfinite(cap)):
        raise ValueError(f"stake_cap must be a positive finite number, got {stake_cap!r}")

    A = algebra.atoms(family)
    P = np.asarray(algebra.payoff_matrix(family, prices), dtype=float)
    m = P.shape[0]
    if P.shape != (m, n):
        raise ValueError(f"payoff matrix {P.shape} != ({m}, {n})")

    # --- LP:  min -g  s.t.  -P s + g <= 0,  -cap <= s <= cap,  g free -------------
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-P, np.ones((m, 1))])
    b_ub = np.zeros(m)
    bounds = [(-cap, cap)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        s = np.asarray(res.x[:n], dtype=float)
        lp_g = float(res.x[n])
        # snap solver dust so the certificate is clean and reproducible
        s[np.abs(s) < max(tol, 1e-12)] = 0.0
        if lp_g <= tol:  # no arbitrage -> the recommended book is to not bet
            s = np.zeros(n)
    else:
        s = np.zeros(n)
        lp_g = 0.0
    if abs(lp_g) < 1e-12:
        lp_g = 0.0  # kill solver -0.0

    profits = P.dot(s)
    g = float(profits.min()) if m else 0.0
    total_abs = float(np.abs(s).sum())
    gpu = g / total_abs if total_abs > max(tol, 1e-12) else 0.0
    arb = bool(g > tol)

    labels = [_target_label(family, t) for t in targets]
    out = {
        "family_id": family.get("family_id"),
        "domain": family.get("domain"),
        "n_atoms": m,
        "n_targets": n,
        "tids": [t.get("tid") for t in targets],
        "target_labels": labels,
        "prices": prices,
        "stake_cap": cap,
        "tol": float(tol),
        "g": g,
        "lp_g": lp_g,
        "stakes": [float(v) for v in s],
        "per_atom_profit": [float(v) for v in profits],
        "atom_labels": [_atom_label(a) for a in A],
        "total_abs_stake": total_abs,
        "g_per_unit_stake": float(gpu),
        "arbitrage": arb,
        "status": str(getattr(res, "message", "")) or ("optimal" if res.success else "failed"),
        "success": bool(res.success),
        "prices_out_of_range": [
            targets[j].get("tid") for j, p in enumerate(prices) if p < 0.0 or p > 1.0
        ],
    }
    out["dollars_per_100"] = dollars_per_100(out)
    out["certificate"] = _certificate(out)
    return out


def _certificate(r):
    """Human-readable arbitrage certificate: the trades, then the guarantee."""
    if not r["success"]:
        return [f"NO CERTIFICATE: LP did not solve ({r['status']})."]
    floor = max(r["tol"], 1e-6)
    lines = []
    for j, s in enumerate(r["stakes"]):
        if abs(s) <= floor:
            continue
        side = "BUY " if s > 0 else "SELL"
        lines.append(
            f"{side} {_fmt_stake(s)} units of {r['target_labels'][j]} "
            f"at their price {_fmt_price(r['prices'][j])}   [{r['tids'][j]}]"
        )
    m = r["n_atoms"]
    if not r["arbitrage"]:
        lp_g = r["lp_g"]
        lines.append(
            f"NO ARBITRAGE: the best bounded book at these prices guarantees "
            f"{lp_g:.6f} <= tol={r['tol']:g}, so no risk-free profit can be extracted "
            f"from these {r['n_targets']} quotes; the quotes are consistent with at least one "
            f"distribution over the {m} atoms. Recommended book: no bets."
        )
        return lines
    best = max(r["per_atom_profit"]) if r["per_atom_profit"] else 0.0
    lines.append(
        f"GUARANTEED PROFIT {r['g']:.4f} per unit stake cap "
        f"(= ${r['dollars_per_100']:.2f} per $100 staked, {r['total_abs_stake']:.2f} units staked): "
        f"this book pays at least {r['g']:.4f} on ALL {m} atoms -- every logically possible "
        f"outcome of this question family -- with a worst atom of {r['g']:.4f} and a best of "
        f"{best:.4f}. The forecaster loses on every outcome, at their own quoted prices."
    )
    return lines


def dollars_per_100(result):
    """The quotable figure: dollars lost per $100 staked = g_per_unit_stake * 100."""
    return float(result.get("g_per_unit_stake", 0.0)) * 100.0


# --------------------------------------------------------------------------- #
# batch run over questions/ x out/forecasts_*.json
# --------------------------------------------------------------------------- #
def _load_forecasts(paths):
    """-> (by_model: {model: {tid: p}}, notes: [str])"""
    by_model, notes = {}, []
    for path in paths:
        stem = pathlib.Path(path).stem
        default_model = stem[len("forecasts_"):] if stem.startswith("forecasts_") else stem
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            notes.append(f"{path}: unreadable ({e})")
            continue
        if isinstance(data, dict):
            data = data.get("forecasts", [])
        if not isinstance(data, list):
            notes.append(f"{path}: expected a JSON list of forecast records")
            continue
        for rec in data:
            if not isinstance(rec, dict):
                notes.append(f"{path}: non-object record skipped")
                continue
            # elicit.py --repeat N writes N draws per tid. Audit draw 0 only: coherence.py's
            # loader does the same, so the two independent auditors are guaranteed to be
            # scoring the identical book. (Without this they kept opposite ends of the run.)
            if rec.get("sample") not in (None, 0):
                continue
            tid, p = rec.get("tid"), rec.get("p")
            model = rec.get("model") or default_model
            if not tid:
                notes.append(f"{path}: record with no tid skipped")
                continue
            try:
                p = float(p)
            except (TypeError, ValueError):
                notes.append(f"{path}: {tid} has non-numeric p={p!r}, treated as missing")
                continue
            if not math.isfinite(p):
                notes.append(f"{path}: {tid} has non-finite p, treated as missing")
                continue
            slot = by_model.setdefault(model, {})
            if tid in slot and abs(slot[tid] - p) > 0:
                notes.append(f"{model}: duplicate {tid} ({slot[tid]} then {p}); using the later")
            slot[tid] = p
    return by_model, notes


def _aggregate(results, n_skipped):
    gpu = [r["g_per_unit_stake"] for r in results]
    arb = [r for r in results if r["arbitrage"]]
    worst = max(results, key=lambda r: (r["g_per_unit_stake"], r["g"])) if results else None
    return {
        "n_families_scored": len(results),
        "n_families_skipped": n_skipped,
        "n_arbitrage": len(arb),
        "frac_arbitrage": (len(arb) / len(results)) if results else 0.0,
        "mean_g_per_unit_stake": (statistics.fmean(gpu) if gpu else 0.0),
        "median_g_per_unit_stake": (statistics.median(gpu) if gpu else 0.0),
        "max_g_per_unit_stake": (max(gpu) if gpu else 0.0),
        "mean_dollars_per_100": (statistics.fmean(gpu) * 100 if gpu else 0.0),
        "median_dollars_per_100": (statistics.median(gpu) * 100 if gpu else 0.0),
        "max_dollars_per_100": (max(gpu) * 100 if gpu else 0.0),
        # total extractable value = sum of guaranteed profits, one book per family at stake_cap
        "total_extractable_value": sum(max(r["g"], 0.0) for r in results),
        "worst_family": (
            {
                "family_id": worst["family_id"],
                "domain": worst["domain"],
                "g": worst["g"],
                "g_per_unit_stake": worst["g_per_unit_stake"],
                "dollars_per_100": worst["dollars_per_100"],
                "total_abs_stake": worst["total_abs_stake"],
                "certificate": worst["certificate"],
            }
            if worst
            else None
        ),
    }


def run(qdir="questions", forecasts_glob="out/forecasts_*.json", stake_cap=1.0, tol=1e-9,
        out_path=OUT_PATH):
    """Audit every (model, family) pair with complete coverage; write out/dutchbook.json."""
    fams, fam_errs = algebra.load_all(qdir)
    for e in fam_errs:
        print(f"  QUESTION ERR {e}")
    paths = sorted(_glob.glob(forecasts_glob))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stake_cap": float(stake_cap),
        "tol": float(tol),
        "lp": "max g s.t. sum_j s_j P[i][j] >= g for all atoms i, -cap <= s_j <= cap (HiGHS)",
        "qdir": qdir,
        "forecasts_glob": forecasts_glob,
        "forecast_files": paths,
        "n_families": len(fams),
        "question_errors": fam_errs,
        "notes": [],
        "results": [],
        "skipped": [],
        "models": {},
    }

    if not paths:
        print(f"dutchbook: no forecast files matched {forecasts_glob!r} -- nothing to audit.")
        print(f"           ({len(fams)} families loaded from {qdir}/)")
        _write(out_path, payload)
        return payload

    by_model, notes = _load_forecasts(paths)
    payload["notes"] = notes
    for note in notes:
        print(f"  NOTE {note}")

    for model in sorted(by_model):
        pm = by_model[model]
        results, skipped = [], []
        for fam in fams:
            tids = [t["tid"] for t in fam["targets"]]
            missing = [t for t in tids if t not in pm]
            if missing:
                skipped.append({"model": model, "family_id": fam["family_id"],
                                "n_missing": len(missing), "missing": missing})
                continue
            try:
                r = dutch_book(fam, [pm[t] for t in tids], stake_cap=stake_cap, tol=tol)
            except Exception as e:  # noqa: BLE001
                payload["notes"].append(f"{model}/{fam['family_id']}: audit failed ({e})")
                print(f"  ERR {model}/{fam['family_id']}: {e}")
                continue
            r["model"] = model
            r["file"] = fam.get("_file")
            results.append(r)
        payload["results"].extend(results)
        payload["skipped"].extend(skipped)
        payload["models"][model] = _aggregate(results, len(skipped))

    _write(out_path, payload)
    _print_summary(payload)
    return payload


def _write(out_path, payload):
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False))


def _print_summary(payload):
    print("\n== Dutch-book audit (de Finetti arbitrage LP) ==")
    print(f"{payload['n_families']} families | {len(payload['forecast_files'])} forecast file(s) "
          f"| stake cap {payload['stake_cap']:g} | tol {payload['tol']:g}")
    hdr = f"{'model':<28}{'fams':>5}{'arb':>5}{'frac':>7}{'mean$':>8}{'med$':>8}{'max$':>8}{'totalG':>9}"
    print(hdr)
    print("-" * len(hdr))
    for model, a in payload["models"].items():
        print(f"{model[:27]:<28}{a['n_families_scored']:>5}{a['n_arbitrage']:>5}"
              f"{a['frac_arbitrage']:>7.2f}{a['mean_dollars_per_100']:>8.2f}"
              f"{a['median_dollars_per_100']:>8.2f}{a['max_dollars_per_100']:>8.2f}"
              f"{a['total_extractable_value']:>9.3f}")
    print("($ columns are dollars guaranteed-lost per $100 staked; totalG sums the "
          "per-family guaranteed profit at the stake cap.)")
    for model, a in payload["models"].items():
        w = a["worst_family"]
        if not w:
            print(f"\n{model}: no families scored.")
            continue
        print(f"\n{model} -- worst family: {w['family_id']} "
              f"(${w['dollars_per_100']:.2f} per $100 staked)")
        for line in w["certificate"]:
            print(f"    {line}")
    if payload["skipped"]:
        print(f"\nSKIPPED {len(payload['skipped'])} (model, family) pair(s) for missing tids:")
        for sk in payload["skipped"][:10]:
            print(f"    {sk['model']} / {sk['family_id']}: missing {sk['n_missing']} "
                  f"tid(s), e.g. {sk['missing'][0]}")
        if len(payload["skipped"]) > 10:
            print(f"    ... and {len(payload['skipped']) - 10} more (see {OUT_PATH})")
    print(f"\nwrote {OUT_PATH}")


# --------------------------------------------------------------------------- #
# self-test on hand-checkable toy families
# --------------------------------------------------------------------------- #
def _ev(text):
    return {"text": text, "resolution": f"Resolves YES iff: {text}",
            "source": "toy self-test (no real source)", "resolve_by": "2027-06-30"}


def _toy_complement():
    """2 base events, quotes P(A)=0.6 AND P(not A)=0.6. Hand check (4 atoms):
        bet A     pays  1[A]     - 0.6
        bet notA  pays  1[not A] - 0.6
    Selling both at cap 1 nets -(0.4) + 0.6 = 0.2 when A is true and 0.6 - 0.4 = 0.2 when
    A is false, i.e. +0.2 on every atom. It is optimal and unique: the two payoffs sum to
    -0.2*(s1+s2) >= -0.4, so min <= -0.1*(s1+s2) <= 0.2 with equality only at s1=s2=-1.
    (Only 2 targets, so algebra.validate -- which requires >=3 -- is not applicable here.)
    """
    return {
        "family_id": "toy_complement", "domain": "toy",
        "base_events": {"A": _ev("Brent crude settles above $95/bbl on 2026-12-31"),
                        "B": _ev("OPEC+ announces a production cut in Q4 2026")},
        "targets": [
            {"tid": "toy_complement.A", "kind": "unconditional", "expr": "A",
             "text": "What is the probability that Brent settles above $95 on 2026-12-31?"},
            {"tid": "toy_complement.notA", "kind": "unconditional", "expr": "not A",
             "text": "What is the probability that Brent does NOT settle above $95 on 2026-12-31?"},
        ],
    }


def _toy_coherent():
    """P(A)=0.5, P(B)=0.4, P(A and B)=0.2, P(A|B)=0.5 -- reproduced exactly by
    q(A,B)=0.2, q(A,~B)=0.3, q(~A,B)=0.2, q(~A,~B)=0.3, so no arbitrage exists."""
    return {
        "family_id": "toy_coherent", "domain": "toy",
        "base_events": {"A": _ev("Brent crude settles above $95/bbl on 2026-12-31"),
                        "B": _ev("OPEC+ announces a production cut in Q4 2026")},
        "targets": [
            {"tid": "toy_coherent.A", "kind": "unconditional", "expr": "A", "text": "P(A)?"},
            {"tid": "toy_coherent.B", "kind": "unconditional", "expr": "B", "text": "P(B)?"},
            {"tid": "toy_coherent.AandB", "kind": "unconditional", "expr": "A and B",
             "text": "P(both)?"},
            {"tid": "toy_coherent.A_g_B", "kind": "conditional", "expr": "A", "given": "B",
             "text": "Assume the OPEC+ cut happens. P(Brent > $95)?"},
        ],
    }


def _toy_disjunction():
    """A, B declared mutually exclusive, quotes P(A)=0.3, P(B)=0.3, P(A or B)=0.9.
    Exclusivity forces P(A or B) = 0.6, so 0.9 is over-priced by 0.3: buy A, buy B,
    sell the disjunction. 3 atoms (TT is excluded)."""
    return {
        "family_id": "toy_disjunction", "domain": "toy",
        "base_events": {"A": _ev("The 2026 award goes to candidate X"),
                        "B": _ev("The 2026 award goes to candidate Y")},
        "constraints": [{"type": "exclusive", "events": ["A", "B"]}],
        "targets": [
            {"tid": "toy_disjunction.A", "kind": "unconditional", "expr": "A", "text": "P(X wins)?"},
            {"tid": "toy_disjunction.B", "kind": "unconditional", "expr": "B", "text": "P(Y wins)?"},
            {"tid": "toy_disjunction.AorB", "kind": "unconditional", "expr": "A or B",
             "text": "P(X or Y wins)?"},
        ],
    }


class _Checks:
    def __init__(self):
        self.failed = 0
        self.n = 0

    def __call__(self, name, ok, detail=""):
        self.n += 1
        self.failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
        return bool(ok)


def _recheck(check, family, prices, r, label):
    """Independent re-substitution: do the RETURNED stakes really guarantee the reported g?"""
    P = np.asarray(algebra.payoff_matrix(family, prices), dtype=float)
    s = np.asarray(r["stakes"], dtype=float)
    profits = P.dot(s)
    ok_atoms = np.allclose(profits, np.asarray(r["per_atom_profit"]), atol=1e-9)
    ok_g = abs(float(profits.min()) - r["g"]) <= 1e-9
    check(f"{label}: stakes re-multiplied through algebra.payoff_matrix reproduce g",
          ok_atoms and ok_g,
          f"min over atoms = {profits.min():.10f} vs reported g = {r['g']:.10f}")
    return profits


def self_test():
    print("dutchbook.py self-test (hand-checkable toy families)")
    check = _Checks()

    # ---- 1. complement violation: P(A)=0.6 and P(not A)=0.6 -------------------
    fam = _toy_complement()
    prices = [0.6, 0.6]
    r = dutch_book(fam, prices, stake_cap=1.0)
    check("complement: arbitrage found", r["arbitrage"] is True, r["status"])
    check("complement: g == 0.2 exactly (1e-6)", abs(r["g"] - 0.2) < 1e-6, f"g={r['g']:.12f}")
    check("complement: optimal book SELLS both bets at cap 1.0",
          all(abs(s - (-1.0)) < 1e-9 for s in r["stakes"]), f"stakes={r['stakes']}")
    spread = max(r["per_atom_profit"]) - min(r["per_atom_profit"])
    check("complement: equal profit on every atom",
          spread < 1e-9 and len(r["per_atom_profit"]) == 4,
          f"{len(r['per_atom_profit'])} atoms, profits={[round(v, 6) for v in r['per_atom_profit']]}")
    check("complement: dollars_per_100 == 10.0 (0.2 profit / 2.0 units staked)",
          abs(dollars_per_100(r) - 10.0) < 1e-6,
          f"${dollars_per_100(r):.4f} per $100, total_abs_stake={r['total_abs_stake']:.2f}")
    _recheck(check, fam, prices, r, "complement")
    print("    certificate:")
    for line in r["certificate"]:
        print(f"      {line}")

    # ---- 2. coherent set: no arbitrage ---------------------------------------
    fam = _toy_coherent()
    prices = [0.5, 0.4, 0.2, 0.5]
    tol = 1e-9
    ok_struct = True
    try:
        algebra.validate(fam)
    except algebra.FamilyError as e:  # pragma: no cover - toy is hand-built to be legal
        ok_struct = False
        print(f"    (toy_coherent failed algebra.validate: {e})")
    check("coherent: toy passes algebra.validate", ok_struct)
    r = dutch_book(fam, prices, stake_cap=1.0, tol=tol)
    check("coherent: g <= tol (no arbitrage)", r["g"] <= tol and not r["arbitrage"],
          f"g={r['g']:.12f}, lp_g={r['lp_g']:.12f}, tol={tol:g}")
    _recheck(check, fam, prices, r, "coherent")
    print(f"    certificate: {r['certificate'][-1]}")

    # ---- 3. over-priced disjunction of exclusive events ----------------------
    fam = _toy_disjunction()
    prices = [0.3, 0.3, 0.9]
    ok_struct = True
    try:
        algebra.validate(fam)
    except algebra.FamilyError as e:  # pragma: no cover
        ok_struct = False
        print(f"    (toy_disjunction failed algebra.validate: {e})")
    check("disjunction: toy passes algebra.validate", ok_struct)
    r = dutch_book(fam, prices, stake_cap=1.0)
    check("disjunction: arbitrage found and g > 0", r["arbitrage"] and r["g"] > 0,
          f"g={r['g']:.6f}, stakes={[round(s, 6) for s in r['stakes']]}, "
          f"${dollars_per_100(r):.2f} per $100")
    profits = _recheck(check, fam, prices, r, "disjunction")
    check("disjunction: hand-substituted min over atoms equals reported g",
          abs(float(profits.min()) - r["g"]) < 1e-9,
          "atom profits " + ", ".join(f"{lab}: {v:+.4f}"
                                      for lab, v in zip(r["atom_labels"], profits)))
    check("disjunction: 3 atoms (exclusivity removed A=T,B=T)", r["n_atoms"] == 3,
          f"atoms={r['atom_labels']}")
    print("    certificate:")
    for line in r["certificate"]:
        print(f"      {line}")

    # ---- 4. extra guards -----------------------------------------------------
    fam = _toy_complement()
    r2 = dutch_book(fam, [0.6, 0.6], stake_cap=2.0)
    check("stake cap scales g linearly (cap 2.0 -> g 0.4, same $ per $100)",
          abs(r2["g"] - 0.4) < 1e-6 and abs(dollars_per_100(r2) - 10.0) < 1e-6,
          f"g={r2['g']:.6f}, ${dollars_per_100(r2):.4f} per $100")
    _recheck(check, fam, [0.6, 0.6], r2, "cap-2.0 complement")
    r3 = dutch_book(_toy_complement(), [0.5, 0.5])
    check("exactly coherent complement (0.5/0.5) gives no arbitrage",
          not r3["arbitrage"] and r3["g"] <= 1e-9, f"g={r3['g']:.12f}")
    _recheck(check, _toy_complement(), [0.5, 0.5], r3, "coherent complement")

    print(f"\n{check.n - check.failed}/{check.n} assertions passed"
          + (f" -- {check.failed} FAILED" if check.failed else " -- ALL PASS"))
    if check.failed:
        raise AssertionError(f"{check.failed} self-test assertion(s) failed")
    return True


if __name__ == "__main__":
    self_test()
    files = sorted(_glob.glob("out/forecasts_*.json"))
    if files:
        print()
        run()
    else:
        print("\nno out/forecasts_*.json yet -- skipping run() (self-test only).")
    sys.exit(0)
