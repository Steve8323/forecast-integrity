"""In-context coherence feedback: the RLCR idea without touching the weights.

RLCR puts a calibration term in the RL reward. This asks whether the same pressure
works purely in the context window: show the model its own answers plus the specific
axiom violations they contain, and let it revise. No training, no gradient.

    arm A  fresh      one question per process, no context           (baseline)
    arm B  joint      all 6 questions in one prompt                  (implicit pressure)
    arm C  feedback   fresh answers + named violations, then revise  (explicit pressure)
    arm D  repair     arm A projected onto the coherent set          (no model involved)

Scored against events that already resolved, so Brier is real. Run: python3 selfrepair.py
"""
import json
import pathlib
import statistics as st

import algebra
import coherence
import dutchbook
import llm

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
MODEL = "claude-sonnet-5"

PROMPT = """You previously estimated these probabilities, each in a separate session with no
memory of the others. They are mutually inconsistent.

YOUR ANSWERS
{answers}

INCONSISTENCIES DETECTED
{violations}

A set of probabilities is coherent when some single distribution over outcomes reproduces
all of them. Yours does not. At your own quoted prices a book can be built that profits
under every possible outcome, worth ${dollars:.2f} per $100 staked.

Revise all {n} estimates so they are mutually consistent. Do not simply move everything to
0.5 -- keep the judgements you are confident in and change the ones that must give. Reply
with ONLY a JSON array of exactly {n} objects, in the SAME NUMBERED ORDER as above:
[{{"n": 1, "p": <float strictly between 0 and 1>}}, ...]"""


def brier(ps, ys):
    return st.fmean((p - y) ** 2 for p, y in zip(ps, ys))


def render(fam, tids, prices):
    """Number every line. The reply is matched back BY POSITION, not by tid -- the model
    invents its own tids when it isn't shown them, and order is what it actually preserves."""
    lines = []
    for i, (t, p) in enumerate(zip(fam["targets"], prices), 1):
        lines.append(f"  [{i}] {t['text'][:110]}  ->  {p:.2f}")
    return "\n".join(lines)


def main():
    fams = json.loads((ROOT / "resolved_families.json").read_text())
    book = {r["tid"]: r["p"] for r in
            json.loads((OUT / f"forecasts_{MODEL}_resolved.json").read_text())
            if isinstance(r.get("p"), (int, float))}
    joint = {r["tid"]: r["p"] for r in
             json.loads((OUT / f"forecasts_{MODEL}_resolved_joint.json").read_text())
             if isinstance(r.get("p"), (int, float))}

    jobs, meta = [], []
    for fam in fams:
        tids = [t["tid"] for t in fam["targets"]]
        prices = [book[t] for t in tids]
        viol = coherence.axiom_violations(fam, prices)
        db = dutchbook.dutch_book(fam, prices)
        vtext = "\n".join(f"  - {v['detail']}" for v in viol) or "  - (none named)"
        jobs.append((PROMPT.format(answers=render(fam, tids, prices), violations=vtext,
                                   dollars=db["dollars_per_100"], n=len(tids)),
                     MODEL, f"selfrepair-{fam['family_id']}"))
        meta.append((fam, tids, prices))

    print(f"asking {len(jobs)} families to revise...")
    outs = llm.call_many(jobs, workers=6)

    rows = {}
    for (fam, tids, prices), raw in zip(meta, outs):
        try:
            arr = llm.extract_json(raw)
            vals = [float(d["p"]) for d in arr]          # matched by position, not tid
            if len(vals) != len(tids):
                raise ValueError(f"got {len(vals)} values for {len(tids)} targets")
            got = {t: v for t, v in zip(tids, vals) if 0 < v < 1}
        except Exception as e:  # noqa: BLE001
            print(f"  parse fail {fam['family_id']}: {e}")
            got = {}
        rows[fam["family_id"]] = [got.get(t) for t in tids]

    arms = {"A fresh": {}, "B joint": {}, "C feedback": {}, "D repair (no model)": {}}
    for (fam, tids, prices) in meta:
        rev = rows[fam["family_id"]]
        arms["A fresh"][fam["family_id"]] = prices
        arms["B joint"][fam["family_id"]] = [joint.get(t) for t in tids]
        arms["C feedback"][fam["family_id"]] = [
            r if r is not None else p for r, p in zip(rev, prices)]
        arms["D repair (no model)"][fam["family_id"]] = coherence.project(fam, prices)

    print(f"\n{'arm':<22} {'Brier':>7} {'skill':>7} {'$/100':>8} {'mean|p-.5|':>11}")
    report = {}
    for name, per in arms.items():
        ps, ys, gs, conf = [], [], [], []
        for (fam, tids, prices) in meta:
            vals = per[fam["family_id"]]
            gs.append(dutchbook.dutch_book(fam, vals)["dollars_per_100"])
            tr = fam["resolved_outcomes"]
            for t, v in zip(tids, vals):
                if tr.get(t) is not None:
                    ps.append(v)
                    ys.append(tr[t])
                    conf.append(abs(v - 0.5))
        b = brier(ps, ys)
        base = brier([st.fmean(ys)] * len(ys), ys)
        report[name] = {"brier": b, "skill": 1 - b / base, "dollars": st.fmean(gs),
                        "confidence": st.fmean(conf), "n": len(ys)}
        print(f"{name:<22} {b:>7.4f} {1 - b / base:>6.1%} {st.fmean(gs):>8.2f} "
              f"{st.fmean(conf):>11.3f}")

    (OUT / "selfrepair.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'selfrepair.json'}")


if __name__ == "__main__":
    main()
