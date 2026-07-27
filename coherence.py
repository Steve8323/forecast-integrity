#!/usr/bin/env python3
"""Coherence auditor: how far is a quoted judgment set from the probability axioms?

Two independent measurements, deliberately kept separate so each checks the other.

1. `incoherence(family, prices)` — a *fit*. The family's state space is the finite
   atom set from `algebra.atoms`. A judgment set is coherent iff some distribution q
   over atoms reproduces every target, so incoherence is the distance to the nearest
   reproducible set:

       min_q  mean_j ( f_j(q) - price_j )^2      s.t.  q >= 0, sum q = 1
       f_j(q) = algebra.implied(q, ev_j, gv_j)

   f_j is linear in q for unconditional targets and a *ratio* for conditional ones,
   so the program is small (atoms <= 8) but non-convex. Solved with SLSQP from a
   deterministic multi-start (uniform + Dirichlet draws), best objective wins.
   `fitted` is the nearest coherent forecast; `project` returns just that.

2. `axiom_violations(family, prices)` — an *audit*, computed straight from the quoted
   numbers with no optimizer in the loop. Each classical identity that the target set
   makes testable is evaluated and every gap reported. If (1) says L2 ~ 0 while (2)
   reports a gap, one of the two is wrong; that is the point of having both.

Axioms checked: complement, partition, conjunction_bound, disjunction, product_rule,
monotonicity, total_probability. Testability is discovered semantically: every `expr`
and `given` is reduced to its truth-signature over the family's atoms, so "A and B"
and "B and A" are the same event, and so are two formulas that the family's declared
constraints force to coincide.

stdlib + numpy + scipy only. Python 3.12 / numpy 1.26 / scipy 1.13.
"""
from __future__ import annotations

import datetime as _dt
import glob as _glob
import itertools
import json
import pathlib
import statistics
import sys
from collections import Counter

import numpy as np
from scipy.optimize import minimize

import algebra

# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------
DENOM_FLOOR = 1e-9      # a conditioning set thinner than this is not divided by
PENALTY = 1e3           # objective value charged to a target whose denominator died
AXIOM_TOL = 1e-9        # a gap below this is floating-point noise, not incoherence
N_DIRICHLET = 12        # deterministic random starts in addition to the uniform one


# ===========================================================================
# 1. the fit
# ===========================================================================
def _masks(family):
    """(atoms, [(ev, gv), ...]) with ev/gv as plain bool lists — algebra's contract."""
    return algebra.target_masks(family)


def _objective(masks, prices):
    """(obj, jac) for mean_j (f_j(q) - price_j)^2.

    Values go through `algebra.implied` so the fit is defined by the shared contract,
    not by a local copy of it. The gradient is analytic — f_j = N_j/D_j with
    N_j = sum_{E and G} q, D_j = sum_G q, so

        df_j/dq_i = ( 1[i in E and G] D_j - N_j 1[i in G] ) / D_j^2

    which matters: finite differences (eps ~ 1.5e-8) leave residuals ~1e-8 on an
    exactly coherent set, and the contract wants L2 < 1e-6 to mean *zero*.
    A conditioning set thinner than DENOM_FLOOR is penalised, never divided by.
    """
    m = len(masks)
    Emat = np.array([[bool(e) and bool(g) for e, g in zip(ev, gv)] for ev, gv in masks],
                    dtype=float)
    Gmat = np.array([[bool(g) for g in gv] for _ev, gv in masks], dtype=float)
    pv = np.asarray(prices, dtype=float)

    def obj(x):
        q = [float(v) for v in x]
        total = 0.0
        for (ev, gv), price in zip(masks, prices):
            denom = 0.0
            for qi, g in zip(q, gv):
                if g:
                    denom += qi
            if denom < DENOM_FLOOR:
                # keep a whisker of slope so SLSQP can climb back out of the wall
                total += PENALTY * (1.0 + (DENOM_FLOOR - denom))
                continue
            f = algebra.implied(q, ev, gv)
            if f is None:                     # algebra's own null-set guard
                total += PENALTY
                continue
            total += (f - price) ** 2
        return total / m

    def jac(x):
        q = np.asarray(x, dtype=float)
        D, N = Gmat @ q, Emat @ q
        g = np.zeros_like(q)
        for j in range(m):
            if D[j] < DENOM_FLOOR:
                g -= PENALTY * Gmat[j]        # push the conditioning mass back up
                continue
            f = N[j] / D[j]
            g += 2.0 * (f - pv[j]) * (Emat[j] * D[j] - N[j] * Gmat[j]) / (D[j] * D[j])
        return g / m

    return obj, jac


def _simplex(x):
    """Clip to [0,1] and renormalise; uniform if the vector has died."""
    q = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    s = q.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full(len(q), 1.0 / len(q))
    return q / s


def _starts(n_atoms, seed):
    """Uniform first, then deterministic Dirichlet draws at three concentrations."""
    rng = np.random.default_rng(seed)
    out = [np.full(n_atoms, 1.0 / n_atoms)]
    if n_atoms > 1:
        alphas = [1.0] * (N_DIRICHLET // 2) + [0.3] * (N_DIRICHLET // 4)
        alphas += [5.0] * (N_DIRICHLET - len(alphas))
        for a in alphas:
            out.append(rng.dirichlet(np.full(n_atoms, a)))
    return out


def _slsqp(obj, jac, x0, n_atoms):
    """SLSQP with sum(q)==1 and bounds [0,1], then one polish restart."""
    cons = ({"type": "eq",
             "fun": lambda x: float(np.sum(x) - 1.0),
             "jac": lambda x: np.ones_like(x)},)
    bnds = [(0.0, 1.0)] * n_atoms
    opts = {"maxiter": 400, "ftol": 1e-16}
    res = minimize(obj, np.asarray(x0, dtype=float), method="SLSQP", jac=jac,
                   bounds=bnds, constraints=cons, options=opts)
    # a second pass from the solution: SLSQP's line search sometimes stalls short of
    # the optimum on the conditional ratios, and this is cheap at <= 8 atoms.
    res2 = minimize(obj, _simplex(res.x), method="SLSQP", jac=jac,
                    bounds=bnds, constraints=cons, options=opts)
    return res2 if res2.fun <= res.fun else res


def incoherence(family, prices, seed=0):
    """Distance from the quoted `prices` to the nearest coherent forecast.

    Returns dict with q, fitted, residuals (prices - fitted), L2 (RMS residual),
    Linf, n_atoms, converged, starts_used.
    """
    A, masks = _masks(family)
    n = len(A)
    prices = [float(p) for p in prices]
    if len(prices) != len(masks):
        raise ValueError(f"{family.get('family_id')}: {len(prices)} prices for "
                         f"{len(masks)} targets")
    obj, jac = _objective(masks, prices)

    starts = _starts(n, seed)
    tried = []
    for x0 in starts:
        try:
            res = _slsqp(obj, jac, x0, n)
        except Exception:                     # noqa: BLE001 - a bad start must not kill the audit
            continue
        q = _simplex(res.x)
        tried.append((obj(q), q, bool(res.success)))   # score the feasible point, not res.fun

    if tried:
        best_val, best, _ok = min(tried, key=lambda t: t[0])
        # honest reading of "converged": some SLSQP run that reported success reached
        # the best objective found. A run can land on the optimum and still report a
        # line-search failure, and that should not be called non-convergence.
        converged = any(ok for val, _q, ok in tried if val <= best_val + 1e-10)
        at_best = sum(1 for val, _q, _ok in tried if val <= best_val + 1e-9)
    else:                                     # every start blew up: fall back to uniform
        best = np.full(n, 1.0 / n)
        best_val, converged, at_best = obj(best), False, 0

    q = best
    fitted = [algebra.implied(q.tolist(), ev, gv) for ev, gv in masks]
    if any(f is None for f in fitted):
        # a null conditioning set survived; nudge toward uniform so the repair is defined
        q = _simplex(0.999999 * q + 1e-6 * np.full(n, 1.0 / n))
        fitted = [algebra.implied(q.tolist(), ev, gv) for ev, gv in masks]
        fitted = [0.0 if f is None else f for f in fitted]
        converged = False

    resid = [p - f for p, f in zip(prices, fitted)]
    L2 = float(np.sqrt(np.mean(np.square(resid)))) if resid else 0.0
    Linf = float(max(abs(r) for r in resid)) if resid else 0.0
    return {
        "family_id": family.get("family_id"),
        "tids": [t["tid"] for t in family["targets"]],
        "q": [float(v) for v in q],
        "fitted": [float(f) for f in fitted],
        "residuals": [float(r) for r in resid],
        "L2": L2,
        "Linf": Linf,
        "objective": float(best_val),
        "n_atoms": n,
        "converged": converged,
        "starts_used": len(starts),
        "starts_at_best": at_best,            # low count = the non-convexity is biting
    }


def project(family, prices, seed=0):
    """The nearest coherent forecast — the deliverable a customer would actually use."""
    return incoherence(family, prices, seed=seed)["fitted"]


# ===========================================================================
# 2. the audit
# ===========================================================================
def _sig(expr, A):
    """Truth-signature of a formula over the family's atoms (its semantic identity)."""
    return tuple(bool(algebra.evaluate(expr, a)) for a in A)


def _flip(sig):
    return tuple(not v for v in sig)


def _and(a, b):
    return tuple(x and y for x, y in zip(a, b))


def _or(a, b):
    return tuple(x or y for x, y in zip(a, b))


def _usable(p):
    return isinstance(p, (int, float)) and not isinstance(p, bool) and np.isfinite(p)


def _entry(axiom, tids, lhs, rhs, detail):
    return {"axiom": axiom, "tids": list(tids), "lhs": float(lhs), "rhs": float(rhs),
            "gap": float(lhs - rhs), "abs_gap": float(abs(lhs - rhs)), "detail": detail}


def _fmt(x):
    return f"{x:.4g}"


def axiom_violations(family, prices):
    """Every classical identity the target set makes testable, evaluated on the quotes.

    Independent of the optimizer. Returns [] when nothing is testable; never raises.
    Sorted by descending |gap|.

    Known blind spot, by design: when two targets share a truth-signature (the same
    event asked twice, possibly via different formulas) only the first is indexed, so
    a disagreement between them is not reported here. That is genuine incoherence and
    `incoherence()` does catch it — an L2 > 0 with an empty violation list is the
    signature of exactly this case, plus any multi-conditional Bayes inconsistency
    that none of the seven named identities spans.
    """
    out, seen = [], set()

    def emit(axiom, tids, lhs, rhs, detail, one_sided=False):
        gap = lhs - rhs
        if one_sided:
            if gap <= AXIOM_TOL:
                return
        elif abs(gap) <= AXIOM_TOL:
            return
        key = (axiom, frozenset(tids))
        if key in seen:
            return
        seen.add(key)
        out.append(_entry(axiom, tids, lhs, rhs, detail))

    try:
        A = algebra.atoms(family)
        targets = family["targets"]
        prices = list(prices)
        if len(prices) != len(targets):
            return []
        full = tuple([True] * len(A))

        # index the quoted targets by semantic signature
        unc, con = [], []           # (idx, tid, sig_e, price) / (..., sig_g, ...)
        for j, t in enumerate(targets):
            if not _usable(prices[j]):
                continue
            try:
                se = _sig(t["expr"], A)
                if t.get("kind") == "conditional":
                    sg = _sig(t["given"], A)
                    con.append((j, t["tid"], se, sg, float(prices[j])))
                else:
                    unc.append((j, t["tid"], se, full, float(prices[j])))
            except Exception:       # noqa: BLE001 - a malformed expr just isn't testable
                continue

        by_sig = {}
        for j, tid, se, _sg, p in unc:
            by_sig.setdefault(se, (tid, p))

        # -- complement: P(E) + P(not E) = 1, unconditional and within a shared given
        for (ja, ta, ea, ga, pa), (jb, tb, eb, gb, pb) in itertools.combinations(unc + con, 2):
            if ga != gb or not any(ga) or eb != _flip_within(ea, eb, ga):
                continue
            kindtxt = "" if ga == full else " given the same conditioning event"
            emit("complement", [ta, tb], pa + pb, 1.0,
                 f"P={_fmt(pa)} and its complement P={_fmt(pb)}{kindtxt} sum to "
                 f"{_fmt(pa + pb)}, must be 1")

        # -- partition: declared exclusive+exhaustive event set sums to 1
        cons = family.get("constraints", []) or []
        excl = [frozenset(c["events"]) for c in cons if c.get("type") == "exclusive"]
        exha = [frozenset(c["events"]) for c in cons if c.get("type") == "exhaustive"]
        for grp in {g for g in excl if g in set(exha)}:
            picked = []
            for e in sorted(grp):
                hit = by_sig.get(_sig(e, A))
                if hit is None:
                    picked = None
                    break
                picked.append(hit)
            if not picked:
                continue
            tot = sum(p for _t, p in picked)
            emit("partition", [t for t, _p in picked], tot, 1.0,
                 f"declared exclusive+exhaustive {{{', '.join(sorted(grp))}}} quoted at "
                 f"{', '.join(_fmt(p) for _t, p in picked)} sums to {_fmt(tot)}, must be 1")

        # -- partition (semantic): >=3 unconditional targets, disjoint, covering
        for k in (3, 4):
            for grp in itertools.combinations(unc, k):
                sigs = [g[2] for g in grp]
                if len({s for s in sigs}) != k:
                    continue
                if any(any(_and(a, b)) for a, b in itertools.combinations(sigs, 2)):
                    continue
                cov = sigs[0]
                for s in sigs[1:]:
                    cov = _or(cov, s)
                if cov != full:
                    continue
                tot = sum(g[4] for g in grp)
                emit("partition", [g[1] for g in grp], tot, 1.0,
                     f"{k} mutually exclusive targets covering the whole state space sum "
                     f"to {_fmt(tot)}, must be 1")

        # -- conjunction bound: P(A and B) <= min(P(A), P(B))
        for (ji, ti, ei, _gi, pi), (jj, tj, ej, _gj, pj) in itertools.combinations(unc, 2):
            if ei == ej:
                continue
            inter = _and(ei, ej)
            for jk, tk, ek, _gk, pk in unc:
                if jk in (ji, jj) or ek != inter:
                    continue
                lo = min(pi, pj)
                emit("conjunction_bound", [tk, ti, tj], pk, lo,
                     f"conjunction quoted at {_fmt(pk)} exceeds min(P={_fmt(pi)}, "
                     f"P={_fmt(pj)})={_fmt(lo)} by {_fmt(pk - lo)}", one_sided=True)

        # -- disjunction: P(A or B) = P(A) + P(B) - P(A and B)
        for (ji, ti, ei, _gi, pi), (jj, tj, ej, _gj, pj) in itertools.combinations(unc, 2):
            if ei == ej:
                continue
            uni, inter = _or(ei, ej), _and(ei, ej)
            for ju, tu, eu, _gu, pu in unc:
                if ju in (ji, jj) or eu != uni:
                    continue
                for jk, tk, ek, _gk, pk in unc:
                    if jk in (ji, jj, ju) or ek != inter:
                        continue
                    rhs = pi + pj - pk
                    emit("disjunction", [tu, ti, tj, tk], pu, rhs,
                         f"P(or)={_fmt(pu)} but P={_fmt(pi)} + P={_fmt(pj)} - "
                         f"P(and)={_fmt(pk)} = {_fmt(rhs)}")

        # -- product rule: P(A and B) = P(A|B) P(B)
        for jc, tc, ec, gc, pc in con:
            g_hit = by_sig.get(gc)
            k_hit = by_sig.get(_and(ec, gc))
            if g_hit is None or k_hit is None or k_hit[0] == g_hit[0]:
                continue
            tb, pb = g_hit
            tk, pk = k_hit
            rhs = pc * pb
            emit("product_rule", [tk, tc, tb], pk, rhs,
                 f"P(and)={_fmt(pk)} but P(cond)={_fmt(pc)} x P(given)={_fmt(pb)} = "
                 f"{_fmt(rhs)}")

        # -- monotonicity: a declared implication forces P(a) <= P(b)
        for c in cons:
            if c.get("type") != "implies":
                continue
            a_hit, b_hit = by_sig.get(_sig(c["a"], A)), by_sig.get(_sig(c["b"], A))
            if a_hit is None or b_hit is None or a_hit[0] == b_hit[0]:
                continue
            emit("monotonicity", [a_hit[0], b_hit[0]], a_hit[1], b_hit[1],
                 f"declared {c['a']} implies {c['b']}, so P={_fmt(a_hit[1])} may not "
                 f"exceed P={_fmt(b_hit[1])} (excess {_fmt(a_hit[1] - b_hit[1])})",
                 one_sided=True)

        # -- total probability: P(A) = P(A|B)P(B) + P(A|not B)P(not B)
        for (j1, t1, e1, g1, p1), (j2, t2, e2, g2, p2) in itertools.combinations(con, 2):
            if e1 != e2 or g2 != _flip(g1) or not any(g1) or not any(g2):
                continue
            a_hit = by_sig.get(e1)
            if a_hit is None:
                continue
            b_hit, nb_hit = by_sig.get(g1), by_sig.get(g2)
            if b_hit is not None:
                tb, pb, src = b_hit[0], b_hit[1], b_hit[0]
            elif nb_hit is not None:
                tb, pb, src = nb_hit[0], 1.0 - nb_hit[1], f"1-{nb_hit[0]}"
            else:
                continue
            rhs = p1 * pb + p2 * (1.0 - pb)
            emit("total_probability", [a_hit[0], t1, t2, tb], a_hit[1], rhs,
                 f"P={_fmt(a_hit[1])} but {_fmt(p1)}x{_fmt(pb)} + {_fmt(p2)}x"
                 f"{_fmt(1 - pb)} = {_fmt(rhs)} (weight from {src})")
    except Exception:                         # noqa: BLE001 - the audit never raises
        pass
    out.sort(key=lambda d: (-d["abs_gap"], d["axiom"], d["tids"]))
    return out


def _flip_within(ea, eb, g):
    """eb's expected signature if it is the complement of ea *inside* g.

    Outside the conditioning set the truth values are irrelevant to the identity, so
    they are copied from eb itself; inside g the complement must hold exactly.
    """
    return tuple((not a) if gi else b for a, b, gi in zip(ea, eb, g))


# ===========================================================================
# 3. batch run
# ===========================================================================
def _load_forecasts(paths):
    """{model: {tid: p}}, plus a list of complaints."""
    models, notes = {}, []
    for p in paths:
        try:
            data = json.loads(pathlib.Path(p).read_text())
        except Exception as e:                # noqa: BLE001
            notes.append(f"{p}: unreadable ({e})")
            continue
        if isinstance(data, dict):
            data = data.get("forecasts", data.get("records", []))
        if not isinstance(data, list):
            notes.append(f"{p}: not a list of forecast records")
            continue
        stem = pathlib.Path(p).stem
        default_model = stem[len("forecasts_"):] if stem.startswith("forecasts_") else stem
        for rec in data:
            if not isinstance(rec, dict):
                continue
            # elicit.py --repeat N writes N draws per tid. Audit draw 0 only: dutchbook.py's
            # loader does the same, so the two independent auditors are guaranteed to be
            # scoring the identical book. (Without this they kept opposite ends of the run.)
            if rec.get("sample") not in (None, 0):
                continue
            tid, val = rec.get("tid"), rec.get("p")
            model = rec.get("model") or default_model
            if not tid or not _usable(val):
                notes.append(f"{p}: dropped record {rec.get('tid')!r} (p={val!r})")
                continue
            book = models.setdefault(model, {})
            if tid in book and abs(book[tid] - float(val)) > 1e-12:
                notes.append(f"{model}: duplicate {tid} ({book[tid]} vs {val}), kept first")
                continue
            book.setdefault(tid, float(val))
    return models, notes


def run(qdir="questions", forecasts_glob="out/forecasts_*.json", seed=0,
        outpath="out/coherence.json"):
    """Audit every (model, family) pair with complete coverage; write out/coherence.json."""
    fams, ferrs = algebra.load_all(qdir)
    for e in ferrs:
        print(f"  family ERR {e}")
    paths = sorted(_glob.glob(forecasts_glob))
    if not paths:
        print(f"coherence: no forecast files matched {forecasts_glob!r} — nothing to audit.")
        return None
    print(f"coherence: {len(fams)} families, {len(paths)} forecast file(s)")
    models, notes = _load_forecasts(paths)
    for n in notes:
        print(f"  note {n}")
    if not models:
        print("coherence: forecast files held no usable records — nothing to audit.")
        return None

    results, skipped = [], []
    for model in sorted(models):
        book = models[model]
        for fam in fams:
            tids = [t["tid"] for t in fam["targets"]]
            missing = [t for t in tids if t not in book]
            if missing:
                skipped.append({"model": model, "family_id": fam["family_id"],
                                "missing": missing})
                continue
            prices = [book[t] for t in tids]
            rep = incoherence(fam, prices, seed=seed)
            rep["model"] = model
            rep["domain"] = fam.get("domain")
            rep["prices"] = prices
            rep["violations"] = axiom_violations(fam, prices)
            repair = [abs(r) for r in rep["residuals"]]
            rep["mean_abs_repair"] = float(np.mean(repair)) if repair else 0.0
            rep["max_abs_repair"] = float(max(repair)) if repair else 0.0
            results.append(rep)

    aggregates = {}
    for model in sorted(models):
        rows = [r for r in results if r["model"] == model]
        vc = Counter(v["axiom"] for r in rows for v in r["violations"])
        repairs = [abs(x) for r in rows for x in r["residuals"]]
        aggregates[model] = {
            "n_families": len(rows),
            "n_targets": sum(len(r["tids"]) for r in rows),
            "n_skipped": sum(1 for s in skipped if s["model"] == model),
            "mean_L2": float(np.mean([r["L2"] for r in rows])) if rows else None,
            "median_L2": float(statistics.median([r["L2"] for r in rows])) if rows else None,
            "max_L2": float(max([r["L2"] for r in rows])) if rows else None,
            "mean_Linf": float(np.mean([r["Linf"] for r in rows])) if rows else None,
            "max_Linf": float(max([r["Linf"] for r in rows])) if rows else None,
            "mean_abs_repair": float(np.mean(repairs)) if repairs else None,
            "max_abs_repair": float(max(repairs)) if repairs else None,
            "n_violations": int(sum(vc.values())),
            "violations_by_axiom": dict(sorted(vc.items())),
            "n_unconverged": sum(1 for r in rows if not r["converged"]),
        }

    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "qdir": qdir,
        "forecasts_glob": forecasts_glob,
        "files": paths,
        "seed": seed,
        "n_families": len(fams),
        "family_errors": ferrs,
        "load_notes": notes,
        "results": results,
        "skipped": skipped,
        "aggregates": aggregates,
    }
    out = pathlib.Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    _print_summary(payload)
    print(f"\ncoherence: wrote {out}")
    return payload


def _print_summary(payload):
    print("\n" + "=" * 88)
    print(f"{'model':<26}{'fams':>5}{'meanL2':>9}{'medL2':>9}{'meanLinf':>10}"
          f"{'repair':>9}{'viol':>6}{'skip':>6}")
    print("-" * 88)
    for model, a in payload["aggregates"].items():
        def f(x):
            return "     -" if x is None else f"{x:.4f}"
        print(f"{model[:26]:<26}{a['n_families']:>5}{f(a['mean_L2']):>9}"
              f"{f(a['median_L2']):>9}{f(a['mean_Linf']):>10}"
              f"{f(a['mean_abs_repair']):>9}{a['n_violations']:>6}{a['n_skipped']:>6}")
    print("-" * 88)
    for model, a in payload["aggregates"].items():
        if a["violations_by_axiom"]:
            bits = ", ".join(f"{k}={v}" for k, v in a["violations_by_axiom"].items())
            print(f"  {model[:26]:<26} {bits}")
        if a["n_unconverged"]:
            print(f"  {model[:26]:<26} SLSQP did not report success on "
                  f"{a['n_unconverged']} famil(ies)")
    worst = sorted(payload["results"], key=lambda r: -r["L2"])[:8]
    if worst:
        print("\nworst families by L2:")
        for r in worst:
            print(f"  {r['L2']:.4f}  Linf={r['Linf']:.4f}  {r['model'][:20]:<20} "
                  f"{r['family_id'][:38]:<38} atoms={r['n_atoms']} "
                  f"viol={len(r['violations'])}")
            for v in r["violations"][:3]:
                print(f"        {v['axiom']:<19} gap={v['gap']:+.4f}  {v['detail']}")
    if payload["skipped"]:
        print(f"\nskipped {len(payload['skipped'])} (model, family) pair(s) for missing tids:")
        for s in payload["skipped"][:12]:
            print(f"  {s['model'][:24]:<24} {s['family_id'][:34]:<34} "
                  f"missing {len(s['missing'])}: {', '.join(s['missing'][:3])}"
                  f"{' ...' if len(s['missing']) > 3 else ''}")
        if len(payload["skipped"]) > 12:
            print(f"  ... and {len(payload['skipped']) - 12} more")


# ===========================================================================
# 4. self-test on hand-checkable toy families
# ===========================================================================
def _ev(txt):
    return {"text": txt, "resolution": "toy", "source": "toy", "resolve_by": "2027-06-30"}


def _toy(fid, targets, constraints=None, names=("A", "B")):
    fam = {"family_id": fid, "domain": "toy",
           "base_events": {n: _ev(f"event {n}") for n in names},
           "targets": []}
    if constraints:
        fam["constraints"] = constraints
    for name, expr, given in targets:
        t = {"tid": f"{fid}.{name}", "text": f"toy {name}", "expr": expr,
             "kind": "conditional" if given else "unconditional"}
        if given:
            t["given"] = given
        fam["targets"].append(t)
    return fam


_FAILS = []


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)
    return ok


def _find(viols, axiom):
    return [v for v in viols if v["axiom"] == axiom]


def selftest():
    print("coherence.py self-test")

    # --- 1. complement violation: P(A)=0.6 and P(not A)=0.6 -----------------
    fam = _toy("toy_comp", [("A", "A", None), ("notA", "not A", None)])
    r = incoherence(fam, [0.6, 0.6])
    _check("complement fit: 4 atoms", r["n_atoms"] == 4, f"n_atoms={r['n_atoms']}")
    _check("complement fit: L2 == 0.1 (1e-6)", abs(r["L2"] - 0.1) < 1e-6,
           f"L2={r['L2']:.10f}")
    _check("complement fit: both fitted at 0.5",
           all(abs(f - 0.5) < 1e-4 for f in r["fitted"]),
           f"fitted={[round(f, 6) for f in r['fitted']]}")
    _check("complement fit: residuals +-0.1",
           all(abs(abs(x) - 0.1) < 1e-4 for x in r["residuals"]),
           f"residuals={[round(x, 6) for x in r['residuals']]}")
    v = _find(axiom_violations(fam, [0.6, 0.6]), "complement")
    _check("complement axiom: reported with gap 0.2",
           len(v) == 1 and abs(v[0]["gap"] - 0.2) < 1e-12,
           f"gap={v[0]['gap'] if v else None}")

    # --- 2. a coherent judgment set must score ~0 ---------------------------
    fam2 = _toy("toy_coh", [("A", "A", None), ("B", "B", None),
                            ("AandB", "A and B", None), ("A_g_B", "A", "B")])
    p2 = [0.5, 0.4, 0.2, 0.5]
    r2 = incoherence(fam2, p2)
    _check("coherent set: L2 < 1e-6", r2["L2"] < 1e-6, f"L2={r2['L2']:.3e}")
    _check("coherent set: q recovers [0.3,0.3,0.2,0.2]",
           max(abs(a - b) for a, b in zip(r2["q"], [0.3, 0.3, 0.2, 0.2])) < 1e-4,
           f"q={[round(x, 6) for x in r2['q']]}")
    _check("coherent set: no axiom violations", axiom_violations(fam2, p2) == [],
           f"{axiom_violations(fam2, p2)}")

    # --- 3. product-rule violation -----------------------------------------
    fam3 = _toy("toy_prod", [("B", "B", None), ("A_g_B", "A", "B"),
                             ("AandB", "A and B", None)])
    p3 = [0.5, 0.8, 0.1]
    v3 = _find(axiom_violations(fam3, p3), "product_rule")
    _check("product rule: flagged with |gap| 0.3",
           len(v3) == 1 and abs(v3[0]["abs_gap"] - 0.3) < 1e-12,
           f"lhs={v3[0]['lhs'] if v3 else None} rhs={v3[0]['rhs'] if v3 else None} "
           f"gap={v3[0]['gap'] if v3 else None}")
    r3 = incoherence(fam3, p3)
    _check("product rule: fit is incoherent (L2 > 0.01)", r3["L2"] > 0.01,
           f"L2={r3['L2']:.4f} fitted={[round(f, 4) for f in r3['fitted']]}")

    # --- 4. conjunction bound ----------------------------------------------
    fam4 = _toy("toy_conj", [("A", "A", None), ("B", "B", None),
                             ("AandB", "B and A", None)])   # reversed order on purpose
    v4 = _find(axiom_violations(fam4, [0.3, 0.4, 0.5]), "conjunction_bound")
    _check("conjunction bound: flagged with gap 0.2 (expr order swapped)",
           len(v4) == 1 and abs(v4[0]["gap"] - 0.2) < 1e-12,
           f"gap={v4[0]['gap'] if v4 else None}")
    _check("conjunction bound: satisfied set is clean",
           _find(axiom_violations(fam4, [0.3, 0.4, 0.2]), "conjunction_bound") == [])

    # --- 5. disjunction identity -------------------------------------------
    fam5 = _toy("toy_disj", [("A", "A", None), ("B", "B", None),
                             ("AandB", "A and B", None), ("AorB", "A or B", None)])
    v5 = _find(axiom_violations(fam5, [0.5, 0.4, 0.2, 0.8]), "disjunction")
    _check("disjunction: flagged with gap +0.1",
           len(v5) == 1 and abs(v5[0]["gap"] - 0.1) < 1e-12,
           f"gap={v5[0]['gap'] if v5 else None}")
    _check("disjunction: coherent quote 0.7 is clean",
           _find(axiom_violations(fam5, [0.5, 0.4, 0.2, 0.7]), "disjunction") == [])

    # --- 6. monotonicity under a declared implication ----------------------
    fam6 = _toy("toy_mono", [("A", "A", None), ("B", "B", None),
                             ("A_g_B", "A", "B")],
                constraints=[{"type": "implies", "a": "A", "b": "B"}])
    r6 = incoherence(fam6, [0.6, 0.4, 0.9])
    _check("monotonicity: implies drops an atom (3 atoms)", r6["n_atoms"] == 3,
           f"n_atoms={r6['n_atoms']}")
    v6 = _find(axiom_violations(fam6, [0.6, 0.4, 0.9]), "monotonicity")
    _check("monotonicity: flagged with gap 0.2",
           len(v6) == 1 and abs(v6[0]["gap"] - 0.2) < 1e-12,
           f"gap={v6[0]['gap'] if v6 else None}")
    _check("monotonicity: P(A)<=P(B) is clean",
           _find(axiom_violations(fam6, [0.3, 0.4, 0.75]), "monotonicity") == [])

    # --- 7. law of total probability ---------------------------------------
    fam7 = _toy("toy_ltp", [("A", "A", None), ("B", "B", None),
                            ("A_g_B", "A", "B"), ("A_g_nB", "A", "not B")])
    v7 = _find(axiom_violations(fam7, [0.5, 0.5, 0.9, 0.9]), "total_probability")
    _check("total probability: flagged with gap -0.4",
           len(v7) == 1 and abs(v7[0]["gap"] + 0.4) < 1e-12,
           f"gap={v7[0]['gap'] if v7 else None}")
    _check("total probability: consistent quote is clean",
           _find(axiom_violations(fam7, [0.9, 0.5, 0.9, 0.9]), "total_probability") == [])

    # --- 8. declared partition ---------------------------------------------
    fam8 = _toy("toy_part", [("A", "A", None), ("B", "B", None), ("AorB", "A or B", None)],
                constraints=[{"type": "exclusive", "events": ["A", "B"]},
                             {"type": "exhaustive", "events": ["A", "B"]}])
    r8 = incoherence(fam8, [0.6, 0.6, 1.0])
    _check("partition: exclusive+exhaustive leaves 2 atoms", r8["n_atoms"] == 2,
           f"n_atoms={r8['n_atoms']}")
    v8 = _find(axiom_violations(fam8, [0.6, 0.6, 1.0]), "partition")
    _check("partition: flagged with gap 0.2",
           len(v8) >= 1 and abs(v8[0]["gap"] - 0.2) < 1e-12,
           f"gaps={[x['gap'] for x in v8]}")

    # --- 9. nothing testable -> [] -----------------------------------------
    fam9 = _toy("toy_none", [("AandB", "A and B", None),
                             ("A_g_B", "A", "B")])
    _check("no testable axiom: returns []", axiom_violations(fam9, [0.3, 0.4]) == [])
    _check("garbage prices never raise",
           axiom_violations(fam9, [None, "x"]) == [] and
           axiom_violations(fam9, [0.3]) == [])

    # --- 10. the repair is itself coherent ---------------------------------
    bad = []
    for fam_, p_ in ((fam, [0.6, 0.6]), (fam3, p3), (fam4, [0.3, 0.4, 0.5]),
                     (fam5, [0.5, 0.4, 0.2, 0.8]), (fam6, [0.6, 0.4, 0.9]),
                     (fam7, [0.5, 0.5, 0.9, 0.9]), (fam8, [0.6, 0.6, 1.0])):
        fitted = project(fam_, p_)
        re = incoherence(fam_, fitted)["L2"]
        if re > 1e-5:
            bad.append((fam_["family_id"], re))
    _check("project() output re-audits as coherent (L2 < 1e-5)", not bad, f"{bad}")

    # --- 11. determinism ----------------------------------------------------
    a1 = incoherence(fam5, [0.5, 0.4, 0.2, 0.8], seed=3)
    a2 = incoherence(fam5, [0.5, 0.4, 0.2, 0.8], seed=3)
    _check("same seed -> identical result", a1["q"] == a2["q"] and a1["L2"] == a2["L2"])

    print(f"\n{'ALL SELF-TESTS PASSED' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    return not _FAILS


if __name__ == "__main__":
    ok = selftest()
    if _glob.glob("out/forecasts_*.json"):
        print()
        run()
    else:
        print("\n(no out/forecasts_*.json yet — skipping run())")
    sys.exit(0 if ok else 1)
