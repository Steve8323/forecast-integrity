"""The controls that decide whether the headline number means anything.

crosscheck.py showed that random uniform quotes are arbitrageable ~100% of the
time. So "N% of families are arbitrageable" is not a finding — almost any noisy
forecaster scores near 100%. The magnitude is the finding, and a magnitude needs
two reference points:

  ceiling   random uniform quotes, ignoring the question text entirely.
            A forecaster no better than this has no probabilistic structure at all.

  floor     the model's OWN repaired (coherent) forecast, re-perturbed by Gaussian
            noise at the model's measured test-retest SD, then re-audited.
            This is the arbitrage you would see from a perfectly coherent forecaster
            whose only defect is that it cannot repeat itself. It is the honest null:
            if the observed arbitrage sits inside this band, the "incoherence" is
            just sampling variance and the headline is dead.

Reported as: observed / floor (how many times the noise floor) and
observed / ceiling (how far toward structureless).

Run: python3 baselines.py [--sd 0.05] [--reps 200]
"""
import argparse
import glob
import json
import pathlib
import statistics as st

import numpy as np

import algebra
import coherence
import dutchbook

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"


def load_forecasts():
    """{(model, arm): {tid: mean_p}} — repeat draws averaged, as scoring.py does."""
    books = {}
    for path in sorted(glob.glob(str(OUT / "forecasts_*.json"))):
        try:
            recs = json.loads(pathlib.Path(path).read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(recs, dict):
            recs = recs.get("forecasts", recs.get("records", []))
        acc = {}
        for r in recs:
            if not isinstance(r, dict):
                continue
            p, tid = r.get("p"), r.get("tid")
            if tid is None or not isinstance(p, (int, float)):
                continue
            key = (r.get("model", pathlib.Path(path).stem), r.get("arm", "fresh"))
            acc.setdefault(key, {}).setdefault(tid, []).append(float(p))
        for key, per in acc.items():
            books.setdefault(key, {}).update({t: st.fmean(v) for t, v in per.items()})
    return books


def retest_sd(model):
    """Mean within-target SD from out/retest_<model>.json, or None.

    The joint arm labels itself "<model>+joint" and has no retest file of its own, so strip
    the arm suffix and reuse the base model's measured SD. Falling back to the 0.05 default
    there would inflate the joint arm's floor and make "joint is indistinguishable from
    noise" easier to claim than the data supports.
    """
    model = model.split("+")[0]
    for path in glob.glob(str(OUT / "retest_*.json")):
        try:
            d = json.loads(pathlib.Path(path).read_text())
        except Exception:  # noqa: BLE001
            continue
        blob = json.dumps(d)
        if model.split("/")[-1] not in blob and model not in path:
            continue
        for k in ("mean_sd", "mean_within_target_sd", "sd_mean"):
            if isinstance(d, dict) and isinstance(d.get(k), (int, float)):
                return float(d[k])
        if isinstance(d, dict):
            sds = [v.get("sd") for v in d.values()
                   if isinstance(v, dict) and isinstance(v.get("sd"), (int, float))]
            if sds:
                return float(st.fmean(sds))
    return None


def audit(fam, prices):
    d = dutchbook.dutch_book(fam, prices)
    return d["g_per_unit_stake"], d["arbitrage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=float, default=None,
                    help="test-retest SD to use for the floor (default: measured, else 0.05)")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    fams, errs = algebra.load_all(str(ROOT / "questions"))
    if not fams:
        print("no families — run the question agents first")
        return
    books = load_forecasts()
    if not books:
        print("no out/forecasts_*.json — run elicit.py first")
        return

    rng = np.random.default_rng(11)
    report = {"n_families": len(fams), "reps": args.reps, "arms": {}}

    for (model, arm), book in sorted(books.items()):
        sd = args.sd if args.sd is not None else (retest_sd(model) or 0.05)
        obs, ceil, floor = [], [], []
        covered = 0
        for fam in fams:
            tids = [t["tid"] for t in fam["targets"]]
            if not all(t in book for t in tids):
                continue
            covered += 1
            prices = [book[t] for t in tids]
            g, _ = audit(fam, prices)
            obs.append(g)

            # ceiling: text-blind random quotes
            for _ in range(max(1, args.reps // len(fams))):
                ceil.append(audit(fam, list(rng.uniform(0.02, 0.98, len(tids))))[0])

            # floor: the model's own repaired forecast + its own retest noise
            rep = coherence.project(fam, prices)
            for _ in range(max(1, args.reps // len(fams))):
                jit = np.clip(np.array(rep) + rng.normal(0, sd, len(rep)), 0.001, 0.999)
                floor.append(audit(fam, list(jit))[0])

        if not obs:
            print(f"{model} [{arm}]: no fully-covered family, skipped")
            continue

        m_obs, m_floor, m_ceil = st.fmean(obs), st.fmean(floor), st.fmean(ceil)
        # one-sided empirical p: how often does pure noise reach the observed level?
        pval = sum(1 for f in floor if f >= m_obs) / max(len(floor), 1)
        row = {
            "model": model, "arm": arm, "families_covered": covered,
            "retest_sd_used": sd,
            "observed_mean_g": m_obs, "observed_median_g": st.median(obs),
            "noise_floor_mean_g": m_floor, "ceiling_mean_g": m_ceil,
            "ratio_to_floor": (m_obs / m_floor) if m_floor > 1e-12 else None,
            "frac_of_ceiling": (m_obs / m_ceil) if m_ceil > 1e-12 else None,
            "p_noise_explains": pval,
            "observed_dollars_per_100": m_obs * 100,
            "floor_dollars_per_100": m_floor * 100,
            "ceiling_dollars_per_100": m_ceil * 100,
        }
        report["arms"][f"{model}|{arm}"] = row
        print(f"\n{model} [{arm}]  {covered} families, retest SD {sd:.3f}")
        print(f"  observed     ${m_obs * 100:6.2f} per $100 staked")
        print(f"  noise floor  ${m_floor * 100:6.2f}   (coherent + own retest noise)")
        print(f"  ceiling      ${m_ceil * 100:6.2f}   (text-blind random quotes)")
        if row["ratio_to_floor"]:
            print(f"  -> {row['ratio_to_floor']:.2f}x the noise floor, "
                  f"{row['frac_of_ceiling']:.0%} of the structureless ceiling, "
                  f"p(noise explains it) = {pval:.3f}")
        if row["ratio_to_floor"] and row["ratio_to_floor"] < 1.5:
            print("  !! NOT clearly above sampling noise — do not headline this number")

    (OUT / "baselines.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'baselines.json'}")


if __name__ == "__main__":
    main()
