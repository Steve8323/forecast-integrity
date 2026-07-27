"""Does forcing coherence make forecasts more ACCURATE, or is it only hygiene?

Everything else in this repo deliberately avoids ground truth. This file is the one
place it is available: families whose events already resolved in 2024-25, so the same
forecast can be scored before and after the coherence repair.

    elicit fresh (one question per process)  ->  incoherent probabilities  ->  Brier
                       |
                       +-- coherence.project()  ->  coherent probabilities  ->  Brier

If repaired Brier < raw Brier, the projection is a product: it makes forecasts better,
not merely tidier. If it is not, coherence is hygiene and should be sold as hygiene.

Called-off conditionals (P(A|B) where B did not occur) have no outcome and are excluded
from scoring, per de Finetti. They still constrain the repair.

Run: python3 repair_test.py [--model claude-sonnet-5] [--longcontext]
"""
import argparse
import json
import pathlib
import shutil
import statistics as st
import subprocess
import sys

import algebra
import coherence

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
SRC = ROOT / "resolved_families.json"
QDIR = ROOT / "_resolved_q"


def brier(ps, ys):
    return st.fmean((p - y) ** 2 for p, y in zip(ps, ys))


def load_families():
    fams = json.loads(SRC.read_text())
    for f in fams:
        algebra.validate(f)
    return fams


def elicit(model, joint=False):
    """Run elicit.py over the resolved families into their own output file."""
    QDIR.mkdir(exist_ok=True)
    shutil.copy(SRC, QDIR / "resolved.json")
    tag = f"{model}_resolved" + ("_joint" if joint else "")
    out = OUT / f"forecasts_{tag.replace('/', '_')}.json"
    cmd = [sys.executable, "elicit.py", "--model", model, "--qdir", str(QDIR),
           "--out", str(out), "--workers", "8"]
    if joint:
        cmd.append("--joint")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=3600)
    print((proc.stdout or "")[-700:])
    if proc.returncode != 0:
        print((proc.stderr or "")[-1200:])
        raise SystemExit(f"elicit failed for {tag}")
    recs = json.loads(out.read_text())
    return {r["tid"]: r["p"] for r in recs
            if isinstance(r.get("p"), (int, float)) and r.get("sample") in (None, 0)}


def score(fams, book, label):
    """Brier before and after repair, over targets with a defined outcome."""
    raw_p, rep_p, ys, per_family = [], [], [], []
    for fam in fams:
        tids = [t["tid"] for t in fam["targets"]]
        if not all(t in book for t in tids):
            print(f"  skip {fam['family_id']}: incomplete coverage")
            continue
        prices = [book[t] for t in tids]
        repaired = coherence.project(fam, prices)
        truth = fam["resolved_outcomes"]
        keep = [(p, r, truth[t]) for t, p, r in zip(tids, prices, repaired)
                if truth.get(t) is not None]
        if not keep:
            continue
        fr = [k[0] for k in keep]
        fp = [k[1] for k in keep]
        fy = [k[2] for k in keep]
        raw_p += fr
        rep_p += fp
        ys += fy
        per_family.append({
            "family_id": fam["family_id"], "n_scored": len(keep),
            "brier_raw": brier(fr, fy), "brier_repaired": brier(fp, fy),
            "arbitrage_raw": None,
        })
    if not ys:
        print(f"{label}: nothing scorable")
        return None
    b_raw, b_rep = brier(raw_p, ys), brier(rep_p, ys)
    base = st.fmean(ys)
    b_base = brier([base] * len(ys), ys)
    improved = sum(1 for f in per_family if f["brier_repaired"] < f["brier_raw"])
    res = {
        "label": label, "n_targets": len(ys), "n_families": len(per_family),
        "base_rate": base,
        "brier_raw": b_raw, "brier_repaired": b_rep, "brier_baserate": b_base,
        "delta": b_rep - b_raw,
        "pct_change": (b_rep - b_raw) / b_raw * 100 if b_raw else 0.0,
        "families_improved": improved, "families_worsened": len(per_family) - improved,
        "skill_raw": 1 - b_raw / b_base if b_base else 0.0,
        "skill_repaired": 1 - b_rep / b_base if b_base else 0.0,
        "per_family": per_family,
    }
    arrow = "BETTER" if res["delta"] < 0 else "WORSE"
    print(f"\n{label}: {len(ys)} scorable targets across {len(per_family)} families "
          f"(base rate {base:.2f})")
    print(f"  Brier raw       {b_raw:.4f}")
    print(f"  Brier repaired  {b_rep:.4f}   -> {arrow} by {abs(res['delta']):.4f} "
          f"({res['pct_change']:+.1f}%)")
    print(f"  Brier base-rate {b_base:.4f}")
    print(f"  families improved by repair: {improved}/{len(per_family)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--joint", action="store_true", help="also run the joint-context arm")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit("resolved_families.json not written yet")
    fams = load_families()
    n_t = sum(len(f["targets"]) for f in fams)
    n_def = sum(1 for f in fams for v in f["resolved_outcomes"].values() if v is not None)
    print(f"{len(fams)} resolved families, {n_t} targets, {n_def} with a defined outcome")

    report = {"model": args.model, "arms": []}
    book = elicit(args.model)
    r = score(fams, book, f"{args.model} fresh")
    if r:
        report["arms"].append(r)
    if args.joint:
        jb = elicit(args.model, joint=True)
        rj = score(fams, jb, f"{args.model} joint")
        if rj:
            report["arms"].append(rj)

    (OUT / "repair_test.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'repair_test.json'}")


if __name__ == "__main__":
    main()
