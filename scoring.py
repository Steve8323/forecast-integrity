"""Proper scoring of committed forecasts, once outcomes land.

A track record is worth nothing without three things, and this module supplies
exactly those three:

1. **A proper scoring rule.** Brier and log score. Both are minimized only by
   reporting your true belief, so they cannot be gamed by hedging or by
   confident-sounding extremes.
2. **A decomposition.** A single Brier number hides whether a forecaster is
   *calibrated* (says 30% when 30% of such things happen) or merely
   *discriminating* (separates the yeses from the nos). Murphy's identity splits
   it: ``brier = reliability - resolution + uncertainty``. Reliability is a
   penalty (lower is better), resolution is a credit (higher is better), and
   uncertainty is the question set's own difficulty -- nothing the forecaster did.
3. **A baseline.** Brier skill score against the base-rate ("climatology")
   forecast. A model that says 12% on everything in a set where 12% happen has a
   fine Brier and zero skill. Skill <= 0 means "no evidence of skill", and saying
   so is the point of the module.

Plus the one curve that actually matters for a long-horizon product:
`by_horizon` bins forecasts by the number of days between when the question was
asked and when it must resolve, and reports Brier and skill per bucket. Skill
that is real at 30 days and gone at 365 days is the normal finding, and it is
invisible in any pooled number.

Record shape: ``{"tid","model","p","rationale","asked_utc"}`` per SCHEMA.md.
Optional fields this module also honors, so upstream elicitation arms need no
changes here:

* ``"sample"`` -- repeat draws of the same target. Draws for one (model, arm, tid)
  are averaged into a single forecast point before scoring, because N correlated
  draws of one question are one question, not N. Treating them as N would shrink
  every interval by sqrt(N) and fake precision.
* ``"arm"`` -- e.g. ``"fresh"`` vs ``"joint"``. Arms are scored separately;
  pooling them would average away the very difference they exist to measure.
  Missing ``arm`` is treated as ``"fresh"``.

`run()` refuses to invent outcomes. Every question in this repo is deliberately
about the future, so the honest state of this module today is ARMED, NOT RUNNABLE.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from datetime import date, datetime, timezone

import numpy as np
from scipy.stats import norm

ROOT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_BUCKETS = (30, 90, 180, 365)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ScoringError(ValueError):
    """Raised on malformed probabilities/outcomes."""


# --------------------------------------------------------------------------- #
# scalar / vector scoring rules
# --------------------------------------------------------------------------- #
def _prep(ps, ys):
    p = np.atleast_1d(np.asarray(ps, dtype=float))
    y = np.atleast_1d(np.asarray(ys, dtype=float))
    if p.shape != y.shape:
        raise ScoringError(f"shape mismatch: ps{p.shape} vs ys{y.shape}")
    if p.size == 0:
        raise ScoringError("no forecasts to score")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ScoringError("probabilities must be finite and in [0,1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ScoringError("outcomes must be 0 or 1 (no partial credit, no NaN)")
    return p, y


def brier(p, y):
    """Brier score (mean squared error), a strictly proper loss. Lower is better.

    Accepts scalars or arrays; returns the mean over all elements, so a scalar
    call returns that single ``(p - y)**2``. Range [0, 1]; always-0.5 scores 0.25.
    """
    pp, yy = _prep(p, y)
    return float(np.mean((pp - yy) ** 2))


def log_score(p, y, eps=1e-6):
    """Logarithmic score reported as a POSITIVE loss, in nats. Lower is better.

    ``-mean[ y*log(p) + (1-y)*log(1-p) ]`` with ``p`` clipped to
    ``[eps, 1-eps]``. Without the clip a single confident miss is infinite, which
    makes the metric hostage to one question; with it, the worst possible
    per-question loss is ``-log(eps)`` (13.8 nats at the default). Log score is
    proper and punishes overconfidence far harder than Brier, which is exactly
    why it is reported alongside it.
    """
    if not (0.0 < eps < 0.5):
        raise ScoringError("eps must be in (0, 0.5)")
    pp, yy = _prep(p, y)
    pc = np.clip(pp, eps, 1.0 - eps)
    return float(-np.mean(yy * np.log(pc) + (1.0 - yy) * np.log(1.0 - pc)))


def _bin_of(p, bins):
    """Bin index for each p: ``[k/bins, (k+1)/bins)``, last bin closed at 1."""
    idx = np.floor(np.asarray(p, dtype=float) * bins).astype(int)
    return np.clip(idx, 0, bins - 1)


# --------------------------------------------------------------------------- #
# Murphy decomposition
# --------------------------------------------------------------------------- #
def murphy(ps, ys, bins=10):
    """Murphy's calibration/refinement decomposition of the Brier score.

    Returns ``{"brier","reliability","resolution","uncertainty","identity_gap",...}``
    where the identity

        brier == reliability - resolution + uncertainty

    holds **exactly** (to floating point, checked at 1e-9 in the self-test) for
    the BINNED estimator. That is the only version for which it can hold: the
    identity is a statement about a forecaster whose forecast is constant within
    each bin, so ``report["brier"]`` is the binned Brier -- the score you would
    have gotten had you rounded every probability to its bin's mean probability.

    The true, unbinned Brier is also returned as ``brier_raw``, and
    ``binning_bias = brier_raw - brier`` is the price of binning. It is small when
    bins are narrow relative to how much the forecaster's p varies inside them,
    and reporting it means the exact identity never has to be paid for with a
    silent lie about what was measured.

    Terms:
      * ``reliability``  = sum_k w_k (p_k - o_k)^2   -- miscalibration PENALTY, 0 is perfect.
      * ``resolution``   = sum_k w_k (o_k - o)^2     -- CREDIT for separating outcomes.
      * ``uncertainty``  = o(1 - o)                  -- the question set's difficulty,
                                                       not an achievement of the forecaster.
    with w_k the fraction of forecasts in bin k, p_k / o_k the bin's mean forecast
    and observed frequency, and o the overall base rate.

    Both reliability and resolution are biased upward at small n (a bin with 3
    questions has noisy o_k); with 10 bins the bias in reliability is roughly
    ``(1/N) * sum_k p_k(1-p_k)``, ~1e-4 at N=20000. Do not read reliability at
    N=40 as calibration.
    """
    p, y = _prep(ps, ys)
    bins = int(bins)
    if bins < 1:
        raise ScoringError("bins must be >= 1")
    n = p.size
    obar = float(y.mean())
    idx = _bin_of(p, bins)

    rel = res = 0.0
    pbar_of = np.empty(n, dtype=float)
    used = 0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        used += 1
        w = nk / n
        pk = float(p[m].mean())
        ok = float(y[m].mean())
        rel += w * (pk - ok) ** 2
        res += w * (ok - obar) ** 2
        pbar_of[m] = pk
    unc = obar * (1.0 - obar)
    brier_binned = float(np.mean((pbar_of - y) ** 2))
    brier_raw = float(np.mean((p - y) ** 2))
    return {
        "brier": brier_binned,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "identity_gap": float(brier_binned - (rel - res + unc)),
        "brier_raw": brier_raw,
        "binning_bias": float(brier_raw - brier_binned),
        "n": int(n),
        "bins": bins,
        "bins_occupied": int(used),
        "base_rate": obar,
        "mean_p": float(p.mean()),
        "identity": "brier == reliability - resolution + uncertainty (binned estimator)",
    }


# --------------------------------------------------------------------------- #
# reliability curve
# --------------------------------------------------------------------------- #
def _wilson(k, n, z):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) interval because it stays inside [0,1] and
    does not collapse to zero width when a bin is all-yes or all-no -- both of
    which happen constantly in the extreme bins of a reliability curve.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    center = (ph + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def reliability_curve(ps, ys, bins=10, conf=0.95):
    """Per-bin calibration table. Empty bins are OMITTED, not zero-filled.

    Each entry: ``{"lo","hi","center","mean_p","obs_freq","n","wilson_lo","wilson_hi"}``
    where ``lo``/``hi`` are the bin edges, ``center`` the edge midpoint,
    ``mean_p`` the mean forecast inside the bin, and ``obs_freq`` the observed
    frequency. Perfect calibration is ``obs_freq == mean_p`` in every bin.

    A bin is only evidence of miscalibration if ``mean_p`` falls outside
    ``[wilson_lo, wilson_hi]``; a curve that wanders far but whose intervals all
    cover the diagonal is a curve with too little data, not a miscalibrated
    forecaster. ``conf`` is the two-sided coverage (default 0.95).
    """
    p, y = _prep(ps, ys)
    bins = int(bins)
    if bins < 1:
        raise ScoringError("bins must be >= 1")
    if not (0.0 < conf < 1.0):
        raise ScoringError("conf must be in (0,1)")
    z = float(norm.ppf(0.5 + conf / 2.0))
    idx = _bin_of(p, bins)
    out = []
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue  # omit empty bins
        lo, hi = k / bins, (k + 1) / bins
        kk = float(y[m].sum())
        wlo, whi = _wilson(kk, nk, z)
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "center": (lo + hi) / 2.0,
                "mean_p": float(p[m].mean()),
                "obs_freq": kk / nk,
                "n": nk,
                "wilson_lo": wlo,
                "wilson_hi": whi,
                "covers_mean_p": bool(wlo <= float(p[m].mean()) <= whi),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# skill vs baseline
# --------------------------------------------------------------------------- #
def brier_skill(ps, ys, base_rate=None):
    """Brier skill score: ``1 - BS / BS_ref`` against the base-rate climatology.

    The reference forecaster ignores every question and quotes ``base_rate`` on
    all of them; ``base_rate`` defaults to the realized frequency ``mean(ys)``,
    which is the *hardest* version of the baseline (it has hindsight the
    forecaster did not). 1.0 is perfect, 0.0 is "no better than knowing nothing
    but the base rate", negative is worse than that.

    Returns ``None`` when the reference is unbeatable-by-construction
    (``BS_ref == 0``: every outcome identical and the base rate exactly equal to
    it), because the ratio is then undefined and any number returned would be an
    artifact.
    """
    p, y = _prep(ps, ys)
    br = float(y.mean()) if base_rate is None else float(base_rate)
    if not math.isfinite(br) or not (0.0 <= br <= 1.0):
        raise ScoringError(f"base_rate must be in [0,1], got {base_rate!r}")
    bs = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((br - y) ** 2))
    if bs_ref <= 0.0:
        return None
    return 1.0 - bs / bs_ref


# --------------------------------------------------------------------------- #
# horizon decay
# --------------------------------------------------------------------------- #
def _parse_dt(s):
    """Parse an ISO-8601 datetime or date into an aware UTC datetime. None on failure."""
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if isinstance(s, date):
        return datetime(s.year, s.month, s.day, tzinfo=timezone.utc)
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        try:
            dt = datetime.fromisoformat(t[:10])
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_families(qdir="questions"):
    """Families from ``questions/*.json`` via algebra.load_all. ([], [err]) on failure."""
    try:
        import algebra
    except Exception as e:  # noqa: BLE001
        return [], [f"algebra unimportable: {e}"]
    p = pathlib.Path(qdir)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        return [], [f"no question dir {p}"]
    try:
        return algebra.load_all(str(p))
    except Exception as e:  # noqa: BLE001
        return [], [f"load_all failed: {e}"]


def resolve_by_map(families):
    """``tid -> resolve_by`` (ISO date string).

    A target's resolve_by is the LATEST resolve_by among the base events its
    ``expr``/``given`` mention -- the target is not settled until every event it
    depends on is settled. Falls back to the family's latest date if no event
    name can be matched.
    """
    out = {}
    for fam in families or []:
        events = fam.get("base_events", {}) or {}
        fam_dates = [ev.get("resolve_by") for ev in events.values() if ev.get("resolve_by")]
        fam_latest = max(fam_dates) if fam_dates else None
        for t in fam.get("targets", []) or []:
            names = set(_NAME.findall(t.get("expr", ""))) | set(_NAME.findall(t.get("given", "") or ""))
            ds = [events[n]["resolve_by"] for n in names if n in events and events[n].get("resolve_by")]
            rb = max(ds) if ds else fam_latest
            if rb:
                out[t["tid"]] = rb
    return out


def bucket_labels(buckets_days=DEFAULT_BUCKETS):
    """Ordered bucket labels for ``buckets_days``, e.g. 0-30d .. 365+d."""
    b = [int(x) for x in buckets_days]
    if not b or any(x <= 0 for x in b) or any(y <= x for x, y in zip(b, b[1:])):
        raise ScoringError(f"buckets_days must be positive and increasing, got {buckets_days!r}")
    labs, prev = [], 0
    for x in b:
        labs.append(f"{prev}-{x}d")
        prev = x
    labs.append(f"{prev}+d")
    return labs


def _bucket_for(days, buckets_days):
    if days is None or not math.isfinite(days):
        return None
    if days < 0:
        return "<0d"
    prev = 0
    for x in buckets_days:
        if days <= x:
            return f"{prev}-{int(x)}d"
        prev = int(x)
    return f"{prev}+d"


def horizon_days(record, rb_map):
    """Days from ``asked_utc`` to the target's ``resolve_by``. None if unknown.

    A record may carry its own ``resolve_by``; otherwise it is looked up by tid
    from the question files. This is the x-axis of the skill-decay curve, so it
    is deliberately the *elicitation-to-resolution* gap and not "days until now".
    """
    asked = _parse_dt(record.get("asked_utc"))
    rb = record.get("resolve_by") or rb_map.get(str(record.get("tid")))
    end = _parse_dt(rb)
    if asked is None or end is None:
        return None
    return (end - asked).total_seconds() / 86400.0


def by_horizon(records, resolutions, buckets_days=DEFAULT_BUCKETS, families=None, bins=10):
    """Brier / skill / n per horizon bucket -- the skill-vs-horizon decay curve.

    Horizon is the number of days between a forecast's ``asked_utc`` and its
    event's ``resolve_by`` (from ``questions/*.json``). Buckets are half-open
    ``(prev, x]`` day ranges built from ``buckets_days``, plus a final ``365+d``
    open bucket and a ``<0d`` bucket for records asked after their own deadline
    (which is a data bug, surfaced rather than dropped).

    Per bucket: ``n``, ``base_rate``, ``mean_p``, ``brier``, ``log_score``,
    ``skill`` (vs that bucket's own base rate) and ``skill_vs_global`` (vs the
    base rate over all scored records). Both are reported because they answer
    different questions: the within-bucket baseline is the fair opponent at that
    horizon, while the global baseline shows whether long-horizon "skill" is just
    the long-horizon questions having a friendlier base rate.

    Empty buckets are omitted. Anything whose horizon cannot be determined lands
    in ``unknown_horizon`` with a count, never silently in a bucket.
    """
    buckets_days = tuple(int(x) for x in buckets_days)
    bucket_labels(buckets_days)  # validates ordering
    if families is None:
        families, _ = load_families()
    rb_map = resolve_by_map(families)

    rows, unknown, unresolved = [], 0, 0
    for r in records or []:
        tid = str(r.get("tid"))
        if tid not in resolutions or resolutions[tid] is None:
            unresolved += 1
            continue
        d = horizon_days(r, rb_map)
        if d is None:
            unknown += 1
            continue
        rows.append((d, float(r["p"]), float(resolutions[tid])))

    out = {
        "buckets_days": list(buckets_days),
        "n_scored": len(rows),
        "unknown_horizon": unknown,
        "unresolved": unresolved,
        "buckets": {},
    }
    if not rows:
        out["note"] = "no resolved records with a determinable horizon"
        return out

    d_all = np.array([r[0] for r in rows])
    p_all = np.array([r[1] for r in rows])
    y_all = np.array([r[2] for r in rows])
    global_br = float(y_all.mean())
    out["global_base_rate"] = global_br
    out["horizon_days_median"] = float(np.median(d_all))
    out["horizon_days_min"] = float(d_all.min())
    out["horizon_days_max"] = float(d_all.max())

    labels = ["<0d"] + bucket_labels(buckets_days)
    assigned = [_bucket_for(d, buckets_days) for d in d_all]
    for lab in labels:
        m = np.array([a == lab for a in assigned], dtype=bool)
        nk = int(m.sum())
        if nk == 0:
            continue
        p, y = p_all[m], y_all[m]
        row = {
            "n": nk,
            "base_rate": float(y.mean()),
            "mean_p": float(p.mean()),
            "median_horizon_days": float(np.median(d_all[m])),
            "brier": brier(p, y),
            "log_score": log_score(p, y),
            "skill": brier_skill(p, y),
            "skill_vs_global": brier_skill(p, y, base_rate=global_br),
        }
        if nk < 20:
            row["warning"] = f"n={nk}: brier and skill here are noise-dominated"
        out["buckets"][lab] = row
    return out


# --------------------------------------------------------------------------- #
# full report
# --------------------------------------------------------------------------- #
def _arm(r):
    a = r.get("arm")
    return str(a) if a else "fresh"


def collapse_repeats(records):
    """Average repeat draws so each (model, arm, tid) is ONE forecast point.

    With ``--repeat N`` the same target is asked N times. Those draws are N
    samples of one belief about one event, so scoring them as N independent
    observations would inflate n by N, shrink every interval by sqrt(N), and
    make a 40-question track record look like 200. The mean draw is the natural
    single forecast (it is also the Brier-optimal summary of the draws), and the
    within-target SD is carried along as ``draw_sd`` for reference.

    Returns ``(points, info)``; ``points`` keeps first-seen order.
    """
    order, groups = [], {}
    dropped = 0
    for r in records or []:
        tid, model = r.get("tid"), r.get("model")
        if tid is None or model is None or r.get("p") is None:
            dropped += 1
            continue
        try:
            p = float(r["p"])
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            dropped += 1
            continue
        key = (str(model), _arm(r), str(tid))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((p, r))
    points, sds = [], []
    for key in order:
        draws = groups[key]
        ps = np.array([d[0] for d in draws], dtype=float)
        base = dict(draws[0][1])
        base["p"] = float(ps.mean())
        base["model"], base["arm"], base["tid"] = key[0], key[1], key[2]
        base["n_draws"] = int(ps.size)
        base["draw_sd"] = float(ps.std(ddof=1)) if ps.size > 1 else 0.0
        base.pop("sample", None)
        points.append(base)
        if ps.size > 1:
            sds.append(base["draw_sd"])
    info = {
        "n_records_in": len(records or []),
        "n_points": len(points),
        "n_dropped_malformed": dropped,
        "n_targets_with_repeats": len(sds),
        "mean_within_target_sd": float(np.mean(sds)) if sds else None,
        "arms": sorted({k[1] for k in order}),
        "models": sorted({k[0] for k in order}),
    }
    return points, info


def _block(points, resolutions, families, bins=10, buckets_days=DEFAULT_BUCKETS):
    """Score one homogeneous group of forecast points."""
    ps, ys, scored = [], [], []
    for r in points:
        tid = str(r["tid"])
        y = resolutions.get(tid)
        if y is None:
            continue
        ps.append(float(r["p"]))
        ys.append(float(y))
        scored.append(r)
    blk = {"n": len(scored), "n_committed": len(points), "n_unresolved": len(points) - len(scored)}
    if not scored:
        blk["note"] = "no resolved outcomes in this group - nothing scoreable yet"
        return blk
    blk.update(
        {
            "base_rate": float(np.mean(ys)),
            "mean_p": float(np.mean(ps)),
            "brier": brier(ps, ys),
            "log_score": log_score(ps, ys),
            "brier_skill": brier_skill(ps, ys),
            "murphy": murphy(ps, ys, bins=bins),
            "reliability_curve": reliability_curve(ps, ys, bins=bins),
            "by_horizon": by_horizon(scored, resolutions, buckets_days, families=families, bins=bins),
        }
    )
    sk = blk["brier_skill"]
    if sk is None:
        blk["verdict"] = "skill undefined (degenerate baseline: every outcome identical)"
    elif blk["n"] < 30:
        blk["verdict"] = f"n={blk['n']}: too few resolved questions to claim skill either way"
    elif sk <= 0:
        blk["verdict"] = f"no evidence of skill (skill={sk:+.3f} vs base rate {blk['base_rate']:.3f})"
    else:
        blk["verdict"] = f"skill={sk:+.3f} over the base-rate baseline on n={blk['n']}"
    return blk


def score(records, resolutions, bins=10, buckets_days=DEFAULT_BUCKETS, families=None):
    """Full scoring report.

    ``resolutions`` is ``{tid: 0|1}``. Repeat draws are collapsed per
    `collapse_repeats`; each (model, arm) pair gets its own block, because arms
    exist precisely to be compared and pooling them would erase the comparison.
    ``pooled`` scores every point together and is only meaningful as an
    all-models ensemble view -- it is labelled as such rather than presented as
    "the" score.
    """
    if families is None:
        families, fam_errs = load_families()
    else:
        fam_errs = []
    resolutions = {str(k): (None if v is None else int(v)) for k, v in (resolutions or {}).items()}
    bad = {k: v for k, v in resolutions.items() if v not in (0, 1, None)}
    if bad:
        raise ScoringError(f"resolutions must be 0/1: offending {list(bad)[:5]}")

    points, info = collapse_repeats(records)
    rep = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_rules": ["brier (proper)", "log_score (proper, nats, clipped)"],
        "baseline": "base-rate climatology (realized frequency), see brier_skill",
        "bins": bins,
        "buckets_days": list(buckets_days),
        "inputs": info,
        "n_resolutions": len([v for v in resolutions.values() if v is not None]),
        "family_errors": fam_errs,
    }
    known = {str(r["tid"]) for r in points}
    rep["resolutions_without_forecast"] = sorted(
        t for t, v in resolutions.items() if v is not None and t not in known
    )[:50]
    rep["unresolved_tids"] = sorted(t for t in known if resolutions.get(t) is None)[:50]

    groups = {}
    for r in points:
        groups.setdefault((r["model"], r["arm"]), []).append(r)
    rep["by_model_arm"] = {
        f"{m}|{a}": _block(g, resolutions, families, bins, buckets_days)
        for (m, a), g in sorted(groups.items())
    }
    rep["pooled"] = _block(points, resolutions, families, bins, buckets_days)
    rep["pooled"]["note"] = (
        "pooled across models and arms; an ensemble view, not any single forecaster's record"
    )
    arms = info["arms"]
    if len(arms) > 1:
        cmp_ = {}
        for a in arms:
            g = [r for r in points if r["arm"] == a]
            b = _block(g, resolutions, families, bins, buckets_days)
            cmp_[a] = {k: b.get(k) for k in ("n", "brier", "log_score", "brier_skill")}
        rep["arm_comparison"] = cmp_
    return rep


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _load_json(path):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _glob(pat):
    """Glob a pattern relative to the repo root (cwd-independent), or absolute."""
    p = pathlib.Path(str(pat))
    if p.is_absolute():
        anchor = pathlib.Path(p.anchor)
        return sorted(anchor.glob(str(p.relative_to(anchor))))
    return sorted(ROOT_DIR.glob(str(p)))


def load_records(glob_pat="out/forecasts_*.json"):
    """All forecast records from every matching file, plus per-file counts."""
    recs, files = [], {}
    for f in _glob(glob_pat):
        data = _load_json(f)
        if isinstance(data, dict):
            data = data.get("forecasts", data.get("records", []))
        if not isinstance(data, list):
            files[f.name] = "unreadable"
            continue
        files[f.name] = len(data)
        recs.extend(data)
    return recs, files


def committed_counts():
    """(total leaves committed, {ledger file: n}) from out/ledger_*.json."""
    per, total = {}, 0
    for f in sorted((ROOT_DIR / "out").glob("ledger_*.json")):
        led = _load_json(f)
        if isinstance(led, dict) and isinstance(led.get("n"), int):
            per[f.name] = {"n": led["n"], "root": led.get("root"), "asof_utc": led.get("asof_utc")}
            total += led["n"]
    return total, per


def earliest_resolve_by(families):
    """(iso_date, tid_or_event) with the earliest resolve_by across families."""
    best = None
    for fam in families or []:
        for name, ev in (fam.get("base_events", {}) or {}).items():
            rb = ev.get("resolve_by")
            if rb and (best is None or rb < best[0]):
                best = (rb, f"{fam.get('family_id')}.{name}")
    return best if best else (None, None)


def run(forecasts_glob="out/forecasts_*.json", resolutions_path="out/resolutions.json"):
    """Score if outcomes exist; otherwise report ARMED state honestly and exit 0."""
    records, files = load_records(forecasts_glob)
    families, fam_errs = load_families()
    res = _load_json(resolutions_path)
    committed, per_ledger = committed_counts()

    if not isinstance(res, dict) or not res:
        points, info = collapse_repeats(records)
        rb_iso, rb_who = earliest_resolve_by(families)
        print("=" * 72)
        print("SCORING: ARMED BUT NOT YET RUNNABLE")
        print("=" * 72)
        print(f"  out/{pathlib.Path(resolutions_path).name} is absent or empty - and it should be.")
        print("  Every question in this repo resolves in the future, so there is nothing to")
        print("  score. Fabricating outcomes here is the exact failure this repo exists to")
        print("  prevent, so run() will not do it.")
        print()
        print(f"  forecast files          : {len(files)}  {dict(sorted(files.items())) if files else '{}'}")
        print(f"  forecast records        : {len(records)}")
        print(f"  distinct forecast points: {info['n_points']} "
              f"(model x arm x tid, repeat draws averaged)")
        if info["arms"]:
            print(f"  arms                    : {', '.join(info['arms'])}")
        if info["models"]:
            print(f"  models                  : {', '.join(info['models'])}")
        if info["mean_within_target_sd"] is not None:
            print(f"  repeat draws            : {info['n_targets_with_repeats']} targets, "
                  f"mean within-target SD {info['mean_within_target_sd']:.4f}")
        print(f"  committed leaves        : {committed} across {len(per_ledger)} ledger(s)")
        for name, d in per_ledger.items():
            print(f"      {name}: n={d['n']} root={str(d['root'])[:16]}... asof={d['asof_utc']}")
        if records and not committed:
            print("      WARNING: forecasts exist but NOTHING is committed. Run"
                  " `python3 ledger.py` before any of these resolve, or the")
            print("      track record is unfalsifiable no matter how it scores.")
        n_fam = len(families)
        n_tgt = sum(len(f.get("targets", [])) for f in families)
        print(f"  question families       : {n_fam} ({n_tgt} targets)")
        if rb_iso:
            days = (_parse_dt(rb_iso) - datetime.now(timezone.utc)).total_seconds() / 86400.0
            print(f"  earliest resolve_by     : {rb_iso}  ({rb_who})  in {days:.0f} days")
            print("  -> scoring becomes runnable when that event settles and its outcome is")
            print("     written to out/resolutions.json as {\"<tid>\": 0|1}.")
        else:
            print("  earliest resolve_by     : unknown (no question files loaded yet)")
        for e in fam_errs[:5]:
            print(f"  question-file error     : {e}")
        print("=" * 72)
        return None

    rep = score(records, res, families=families)
    rep["ledgers"] = per_ledger
    rep["forecast_files"] = files
    out_path = ROOT_DIR / "out" / "scoring.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
    print(f"scoring: {rep['inputs']['n_points']} forecast points, "
          f"{rep['n_resolutions']} resolutions")
    for key, blk in rep["by_model_arm"].items():
        if blk.get("n"):
            m = blk["murphy"]
            print(f"  {key}: n={blk['n']} brier={blk['brier']:.4f} log={blk['log_score']:.4f} "
                  f"skill={blk['brier_skill'] if blk['brier_skill'] is None else round(blk['brier_skill'], 4)}")
            print(f"      reliability={m['reliability']:.5f} resolution={m['resolution']:.5f} "
                  f"uncertainty={m['uncertainty']:.5f}")
            print(f"      {blk['verdict']}")
            for lab, row in blk["by_horizon"]["buckets"].items():
                sk = row["skill"]
                print(f"      horizon {lab:>9}: n={row['n']:>4} brier={row['brier']:.4f} "
                      f"skill={'n/a' if sk is None else format(sk, '+.3f')}")
        else:
            print(f"  {key}: {blk.get('note')}")
    print(f"  wrote {out_path.relative_to(ROOT_DIR)}")
    return rep


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _toy_family():
    """Hand-checkable toy family: two events, B implied by A, four targets."""
    return {
        "family_id": "toy_fam",
        "domain": "toy",
        "base_events": {
            "A": {"text": "A happens", "resolution": "r", "source": "s", "resolve_by": "2026-09-01"},
            "B": {"text": "B happens", "resolution": "r", "source": "s", "resolve_by": "2027-06-30"},
        },
        "constraints": [{"type": "implies", "a": "A", "b": "B"}],
        "targets": [
            {"tid": "toy_fam.A", "kind": "unconditional", "text": "P(A)?", "expr": "A"},
            {"tid": "toy_fam.B", "kind": "unconditional", "text": "P(B)?", "expr": "B"},
            {"tid": "toy_fam.AandB", "kind": "unconditional", "text": "P(A and B)?", "expr": "A and B"},
            {"tid": "toy_fam.A_g_B", "kind": "conditional", "text": "P(A|B)?", "expr": "A", "given": "B"},
        ],
    }


def self_test():
    fails = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name} {detail}")
            fails.append(name)

    # ---- scalar sanity ----------------------------------------------------
    check("brier(1,1)==0 and brier(0,1)==1", brier(1.0, 1) == 0.0 and brier(0.0, 1) == 1.0)
    check("brier(0.5,y)==0.25", brier(0.5, 1) == 0.25 and brier(0.5, 0) == 0.25)
    check("log_score is a positive loss, clipped",
          abs(log_score(0.5, 1) - math.log(2)) < 1e-12
          and abs(log_score(0.0, 1) - -math.log(1e-6)) < 1e-9)
    check("log_score punishes a confident miss harder than brier",
          log_score(0.99, 0) / log_score(0.6, 0) > brier(0.99, 0) / brier(0.6, 0))
    for bad, why in [((1.2, 1), "p>1"), ((0.5, 0.5), "y not 0/1"), (([0.1, 0.2], [1]), "shape")]:
        try:
            brier(*bad)
            check(f"rejects {why}", False)
        except ScoringError:
            check(f"rejects {why}", True)

    # ---- calibrated vs overconfident, n=20000 ----------------------------
    rng = np.random.default_rng(20260726)
    N = 20000
    p_cal = rng.uniform(0.0, 1.0, N)
    y = (rng.uniform(0.0, 1.0, N) < p_cal).astype(int)  # y ~ Bernoulli(p): calibrated by construction

    lo = np.clip(p_cal, 1e-9, 1 - 1e-9)
    logit = np.log(lo / (1 - lo))
    p_over = 1.0 / (1.0 + np.exp(-2.5 * logit))  # push toward 0/1: same ranking, worse calibration

    m_cal = murphy(p_cal, y, bins=10)
    m_over = murphy(p_over, y, bins=10)
    rc_cal = reliability_curve(p_cal, y, bins=10)

    print(f"       calibrated : brier={m_cal['brier']:.5f} rel={m_cal['reliability']:.6f} "
          f"res={m_cal['resolution']:.5f} unc={m_cal['uncertainty']:.5f}")
    print(f"       overconfid.: brier={m_over['brier']:.5f} rel={m_over['reliability']:.6f} "
          f"res={m_over['resolution']:.5f} unc={m_over['uncertainty']:.5f}")

    check(f"calibrated reliability ~ 0 ({m_cal['reliability']:.2e} < 5e-3)",
          m_cal["reliability"] < 0.005)
    check(f"calibrated resolution > 0 ({m_cal['resolution']:.4f})", m_cal["resolution"] > 0.0)
    check("calibrated resolution ~ Var(p) = 1/12 for uniform p",
          abs(m_cal["resolution"] - 1 / 12) < 0.01, m_cal["resolution"])
    check(f"Murphy identity holds to 1e-9 (calibrated, gap={m_cal['identity_gap']:.2e})",
          abs(m_cal["identity_gap"]) < 1e-9)
    check(f"Murphy identity holds to 1e-9 (overconfident, gap={m_over['identity_gap']:.2e})",
          abs(m_over["identity_gap"]) < 1e-9)
    check("overconfident reliability materially worse (>10x and >+0.01)",
          m_over["reliability"] > max(10 * m_cal["reliability"], m_cal["reliability"] + 0.01),
          f"{m_over['reliability']:.5f} vs {m_cal['reliability']:.5f}")
    check("overconfident brier WORSE than calibrated",
          m_over["brier"] > m_cal["brier"] + 0.005,
          f"{m_over['brier']:.5f} vs {m_cal['brier']:.5f}")
    check("overconfident raw brier also worse",
          brier(p_over, y) > brier(p_cal, y) + 0.005)
    check("calibrated brier ~ E[p(1-p)] = 1/6 for uniform p",
          abs(brier(p_cal, y) - 1 / 6) < 0.01, brier(p_cal, y))
    check("overconfident log score worse than calibrated",
          log_score(p_over, y) > log_score(p_cal, y))
    check("binning bias is small at 10 bins",
          abs(m_cal["binning_bias"]) < 0.01, m_cal["binning_bias"])
    check("Murphy identity holds for bins=1 and bins=50",
          abs(murphy(p_cal, y, bins=1)["identity_gap"]) < 1e-9
          and abs(murphy(p_cal, y, bins=50)["identity_gap"]) < 1e-9)
    check("bins=1 has zero resolution (a constant forecaster discriminates nothing)",
          murphy(p_cal, y, bins=1)["resolution"] < 1e-12)

    # ---- reliability curve ----------------------------------------------
    check("reliability curve has one entry per occupied bin", len(rc_cal) == 10)
    check("reliability curve n sums to N", sum(r["n"] for r in rc_cal) == N)
    check("calibrated curve: every Wilson interval covers mean_p",
          all(r["covers_mean_p"] for r in rc_cal),
          [round(r["center"], 2) for r in rc_cal if not r["covers_mean_p"]])
    rc_over = reliability_curve(p_over, y, bins=10)
    check("overconfident curve: some Wilson interval EXCLUDES mean_p",
          any(not r["covers_mean_p"] for r in rc_over))
    rc_sparse = reliability_curve([0.05, 0.06, 0.95], [0, 0, 1], bins=10)
    check("empty bins are omitted", len(rc_sparse) == 2 and {r["lo"] for r in rc_sparse} == {0.0, 0.9})
    check("Wilson interval is non-degenerate on an all-yes bin",
          rc_sparse[1]["wilson_lo"] > 0.0 and rc_sparse[1]["wilson_hi"] <= 1.0
          and rc_sparse[1]["wilson_hi"] - rc_sparse[1]["wilson_lo"] > 0.0)
    w_lo, w_hi = _wilson(0, 10, 1.959963985)
    check("Wilson on 0/10 stays in [0,1] and has width",
          w_lo == 0.0 and 0.2 < w_hi < 0.35, (w_lo, w_hi))

    # ---- skill vs baseline ----------------------------------------------
    br = float(y.mean())
    s_self = brier_skill(np.full(N, br), y)
    check(f"skill of the base-rate forecast against itself ~ 0 ({s_self:.2e})", abs(s_self) < 1e-12)
    check("skill of the calibrated forecaster > 0", brier_skill(p_cal, y) > 0.1)
    check("skill of the overconfident forecaster is lower",
          brier_skill(p_over, y) < brier_skill(p_cal, y))
    check("a constant 0.5 forecaster has skill <= 0 on a skewed set",
          brier_skill([0.5] * 4, [0, 0, 0, 1]) <= 0.0)
    check("degenerate baseline returns None", brier_skill([0.3, 0.4], [1, 1], base_rate=1.0) is None)
    check("explicit base_rate is honored",
          abs(brier_skill([0.2] * 4, [0, 0, 0, 1], base_rate=0.2) - 0.0) < 1e-12)

    # ---- horizon buckets on the toy family ------------------------------
    fam = _toy_family()
    try:
        import algebra
        algebra.validate(fam)
        check("toy family validates against algebra.py", True)
    except Exception as e:  # noqa: BLE001
        check("toy family validates against algebra.py", False, str(e))
    rb = resolve_by_map([fam])
    check("target resolve_by = latest of the events it mentions",
          rb["toy_fam.A"] == "2026-09-01" and rb["toy_fam.B"] == "2027-06-30"
          and rb["toy_fam.AandB"] == "2027-06-30" and rb["toy_fam.A_g_B"] == "2027-06-30", rb)
    asked = "2026-07-26T00:00:00+00:00"
    d_a = horizon_days({"tid": "toy_fam.A", "asked_utc": asked}, rb)
    check("horizon 2026-07-26 -> 2026-09-01 is 37 days", abs(d_a - 37.0) < 1e-9, d_a)
    check("bucket labels are ordered and exhaustive",
          bucket_labels() == ["0-30d", "30-90d", "90-180d", "180-365d", "365+d"])
    check("37 days lands in 30-90d", _bucket_for(37.0, DEFAULT_BUCKETS) == "30-90d")
    check("bucket edges are (prev, x] half-open",
          _bucket_for(30.0, DEFAULT_BUCKETS) == "0-30d"
          and _bucket_for(30.1, DEFAULT_BUCKETS) == "30-90d"
          and _bucket_for(400.0, DEFAULT_BUCKETS) == "365+d")
    check("negative horizon is surfaced, not dropped", _bucket_for(-3.0, DEFAULT_BUCKETS) == "<0d")

    # skill should decay with horizon: near questions informative, far ones noise
    n_h = 400
    recs_h, res_h = [], {}
    p_near = rng.uniform(0, 1, n_h)
    y_near = (rng.uniform(0, 1, n_h) < p_near).astype(int)
    y_far = rng.integers(0, 2, n_h)
    for i in range(n_h):
        recs_h.append({"tid": f"near.{i}", "model": "m", "p": float(p_near[i]),
                       "asked_utc": asked, "resolve_by": "2026-08-10"})
        res_h[f"near.{i}"] = int(y_near[i])
        recs_h.append({"tid": f"far.{i}", "model": "m", "p": 0.5 + 0.45 * (rng.uniform() - 0.5) * 2,
                       "asked_utc": asked, "resolve_by": "2027-12-01"})
        res_h[f"far.{i}"] = int(y_far[i])
    bh = by_horizon(recs_h, res_h, families=[])
    check("by_horizon splits into the right buckets",
          set(bh["buckets"]) == {"0-30d", "365+d"}, list(bh["buckets"]))
    check("by_horizon n per bucket is right",
          bh["buckets"]["0-30d"]["n"] == n_h and bh["buckets"]["365+d"]["n"] == n_h)
    check("skill decays with horizon (near > far)",
          bh["buckets"]["0-30d"]["skill"] > bh["buckets"]["365+d"]["skill"],
          (bh["buckets"]["0-30d"]["skill"], bh["buckets"]["365+d"]["skill"]))
    bh_u = by_horizon([{"tid": "x", "model": "m", "p": 0.5}], {"x": 1}, families=[])
    check("unknown horizon is counted, never bucketed", bh_u["unknown_horizon"] == 1 and not bh_u["buckets"])

    # ---- repeat draws and arms ------------------------------------------
    reps = [
        {"tid": "toy_fam.A", "model": "m", "p": 0.20, "sample": 0, "asked_utc": asked},
        {"tid": "toy_fam.A", "model": "m", "p": 0.30, "sample": 1, "asked_utc": asked},
        {"tid": "toy_fam.A", "model": "m", "p": 0.40, "sample": 2, "asked_utc": asked},
        {"tid": "toy_fam.A", "model": "m", "p": 0.90, "arm": "joint", "asked_utc": asked},
    ]
    pts, info = collapse_repeats(reps)
    check("repeat draws collapse to one point per (model, arm, tid)", len(pts) == 2, len(pts))
    check("collapsed p is the mean draw", abs(pts[0]["p"] - 0.30) < 1e-12 and pts[0]["n_draws"] == 3)
    check("within-target SD is carried through",
          abs(pts[0]["draw_sd"] - float(np.std([0.2, 0.3, 0.4], ddof=1))) < 1e-12)
    check("arms are kept separate", info["arms"] == ["fresh", "joint"])
    check("malformed records are dropped, counted",
          collapse_repeats([{"tid": "t", "model": "m", "p": None}])[1]["n_dropped_malformed"] == 1)

    # ---- end-to-end score() on the toy family ---------------------------
    toy_recs, toy_res = [], {}
    for i, t in enumerate(fam["targets"]):
        toy_recs.append({"tid": t["tid"], "model": "toy-model", "p": [0.2, 0.5, 0.15, 0.4][i],
                         "rationale": "toy", "asked_utc": asked})
        toy_res[t["tid"]] = [0, 1, 0, 1][i]
    rep = score(toy_recs, toy_res, bins=5, families=[fam])
    blk = rep["by_model_arm"]["toy-model|fresh"]
    hand_brier = float(np.mean([(0.2 - 0) ** 2, (0.5 - 1) ** 2, (0.15 - 0) ** 2, (0.4 - 1) ** 2]))
    check(f"score() brier matches hand calculation ({hand_brier:.6f})",
          abs(blk["brier"] - hand_brier) < 1e-12, blk["brier"])
    check("score() reports n=4 and the base rate", blk["n"] == 4 and abs(blk["base_rate"] - 0.5) < 1e-12)
    check("score() refuses to claim skill at n=4", "too few" in blk["verdict"], blk["verdict"])
    check("score() Murphy identity holds on the toy",
          abs(blk["murphy"]["identity_gap"]) < 1e-9)
    check("score() fills the horizon buckets from the toy family",
          set(blk["by_horizon"]["buckets"]) == {"30-90d", "180-365d"},
          list(blk["by_horizon"]["buckets"]))
    check("score() is JSON-serializable", isinstance(json.dumps(rep, default=str), str))
    part = score(toy_recs, {"toy_fam.A": 1}, bins=5, families=[fam])
    check("score() tolerates partial resolutions",
          part["by_model_arm"]["toy-model|fresh"]["n"] == 1
          and part["by_model_arm"]["toy-model|fresh"]["n_unresolved"] == 3)
    check("score() lists resolutions with no matching forecast",
          score(toy_recs, {"ghost.tid": 1}, families=[fam])["resolutions_without_forecast"] == ["ghost.tid"])
    try:
        score(toy_recs, {"toy_fam.A": 2}, families=[fam])
        check("score() rejects non-binary resolutions", False)
    except ScoringError:
        check("score() rejects non-binary resolutions", True)

    print()
    if fails:
        print(f"FAIL  scoring.py self-test: {len(fails)} assertion(s) failed: {fails}")
        return False
    print("PASS  scoring.py self-test: proper rules, Murphy identity <1e-9, calibrated vs")
    print("      overconfident separation, Wilson bins, base-rate skill, horizon decay")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score committed forecasts once outcomes land.")
    ap.add_argument("--glob", default="out/forecasts_*.json")
    ap.add_argument("--resolutions", default="out/resolutions.json")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--no-self-test", action="store_true")
    a = ap.parse_args(argv)

    ok = True
    if not a.no_self_test:
        print("scoring.py self-test")
        ok = self_test()
        if a.self_test_only or not ok:
            return 0 if ok else 1
        print()
    run(a.glob, a.resolutions)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
