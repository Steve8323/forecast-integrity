"""End-to-end driver. Order is part of the argument being made.

    validate -> elicit -> COMMIT -> audit -> score -> report

The commit happens before any analysis touches the numbers, because a track record
that was not sealed before the outcome was known is not a track record. Every stage
is a subprocess so one broken module cannot take down the run.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"

MODELS_DEFAULT = ["claude-sonnet-5", "claude-opus-5"]


def stage(name, cmd, optional=False):
    print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True,
                          capture_output=True, timeout=7200)
    tail = (proc.stdout or "")[-4000:]
    print(tail)
    if proc.returncode != 0:
        print(f"-- stderr --\n{(proc.stderr or '')[-2500:]}")
        if not optional:
            print(f"!! {name} FAILED (rc={proc.returncode})")
        return {"stage": name, "ok": False, "secs": round(time.time() - t0, 1)}
    print(f"-- {name} ok in {time.time() - t0:.1f}s")
    return {"stage": name, "ok": True, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="cap targets per model (smoke run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-elicit", action="store_true", help="reuse existing out/forecasts_*.json")
    ap.add_argument("--skip-leakage", action="store_true")
    args = ap.parse_args()

    log = []

    # 0. the question set must be structurally sound before a single call is made
    import algebra
    fams, errs = algebra.load_all(str(ROOT / "questions"))
    n_targets = sum(len(f["targets"]) for f in fams)
    print(f"{len(fams)} families, {n_targets} targets, {len(errs)} validation errors")
    for e in errs:
        print("  ERR", e)
    if not fams:
        sys.exit("no valid families — nothing to run")
    log.append({"stage": "validate", "ok": not errs, "families": len(fams),
                "targets": n_targets, "errors": errs})

    # 1. elicit, one fresh context per target
    if not args.skip_elicit:
        for m in args.models:
            cmd = ["elicit.py", "--model", m, "--workers", str(args.workers)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            log.append(stage(f"elicit {m}", cmd))

    # 2. seal the forecasts before anything analyses them
    log.append(stage("ledger (commit)", ["ledger.py"]))

    # 3. the two independent coherence audits
    log.append(stage("coherence (QP projection)", ["coherence.py"]))
    log.append(stage("dutchbook (LP arbitrage)", ["dutchbook.py"]))

    # 4. contamination
    if not args.skip_leakage:
        cmd = ["leakage.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        log.append(stage("leakage", cmd, optional=True))

    # 5. scoring is armed but cannot run until the future arrives
    log.append(stage("scoring (armed)", ["scoring.py"], optional=True))

    # 6. report
    log.append(stage("report", ["report.py"], optional=True))

    (OUT / "run_log.json").write_text(json.dumps(log, indent=2))
    print("\n" + "=" * 72)
    for s in log:
        print(f"  {'ok ' if s.get('ok') else 'FAIL'}  {s['stage']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
