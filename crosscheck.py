"""Cross-check the two coherence audits against each other.

coherence.py answers "how far is this judgment set from the coherent set?" by
projection. dutchbook.py answers "can I extract a guaranteed profit from it?" by
linear programming. They share no code beyond algebra.py, and de Finetti's theorem
says they must agree on the yes/no question:

    projection residual == 0   <=>   no arbitrage exists

So agreement is a real test of both, not a tautology. Disagreement means one of
them is wrong, and this script is how we would find out.

Run: python3 crosscheck.py [n_trials]
"""
import sys

import numpy as np

import algebra
import coherence
import dutchbook

TOL_COHERENT = 1e-4   # projection L2 below this counts as "coherent"
TOL_ARB = 1e-6        # dutch-book g above this counts as "arbitrage"


def _ev(name, resolve="2027-01-01"):
    return {"text": f"event {name}", "resolution": "r", "source": "s", "resolve_by": resolve}


def families():
    """A spread of structures: plain, implies, 3-way partition, and a conditional-heavy one."""
    base2 = {"A": _ev("A"), "B": _ev("B")}
    base3 = {"A": _ev("A"), "B": _ev("B"), "C": _ev("C")}
    T = lambda tid, expr, given=None: (  # noqa: E731
        {"tid": tid, "kind": "conditional", "text": tid, "expr": expr, "given": given}
        if given else
        {"tid": tid, "kind": "unconditional", "text": tid, "expr": expr})
    return [
        {"family_id": "x_plain", "domain": "t", "base_events": base2, "constraints": [],
         "targets": [T("x_plain.A", "A"), T("x_plain.B", "B"), T("x_plain.AB", "A and B"),
                     T("x_plain.AoB", "A or B")]},
        {"family_id": "x_cond", "domain": "t", "base_events": base2, "constraints": [],
         "targets": [T("x_cond.A", "A"), T("x_cond.B", "B"), T("x_cond.AB", "A and B"),
                     T("x_cond.A_B", "A", "B"), T("x_cond.B_A", "B", "A"),
                     T("x_cond.A_nB", "A", "not B")]},
        {"family_id": "x_impl", "domain": "t", "base_events": base2,
         "constraints": [{"type": "implies", "a": "A", "b": "B"}],
         "targets": [T("x_impl.A", "A"), T("x_impl.B", "B"), T("x_impl.A_B", "A", "B")]},
        {"family_id": "x_part", "domain": "t", "base_events": base3,
         "constraints": [{"type": "exclusive", "events": ["A", "B", "C"]},
                         {"type": "exhaustive", "events": ["A", "B", "C"]}],
         "targets": [T("x_part.A", "A"), T("x_part.B", "B"), T("x_part.C", "C"),
                     T("x_part.AoB", "A or B")]},
        {"family_id": "x_mix", "domain": "t", "base_events": base3, "constraints": [],
         "targets": [T("x_mix.A", "A"), T("x_mix.BC", "B and C"), T("x_mix.AoB", "A or B"),
                     T("x_mix.A_BC", "A", "B and C"), T("x_mix.nA", "not A")]},
    ]


def coherent_prices(fam, rng):
    """Sample a genuinely coherent quote set: draw q on the atom simplex, read off targets."""
    A, masks = algebra.target_masks(fam)
    for _ in range(200):
        q = rng.dirichlet(np.ones(len(A)) * 1.5)
        ps = [algebra.implied(list(q), ev, gv) for ev, gv in masks]
        if all(p is not None and 1e-3 < p < 1 - 1e-3 for p in ps):
            return ps
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(7)
    rows, bad = [], []

    for fam in families():
        algebra.validate(fam)
        nt = len(fam["targets"])

        # arm 1: coherent by construction -> both must say "clean"
        for _ in range(n // 2):
            ps = coherent_prices(fam, rng)
            if ps is None:
                continue
            c = coherence.incoherence(fam, ps)
            d = dutchbook.dutch_book(fam, ps)
            rows.append((fam["family_id"], "coherent", c["L2"], d["g"]))
            if c["L2"] > TOL_COHERENT or d["arbitrage"]:
                bad.append(("coherent-by-construction flagged", fam["family_id"], ps,
                            c["L2"], d["g"]))

        # arm 2: independent uniform quotes -> the two verdicts must still match
        for _ in range(n):
            ps = list(rng.uniform(0.02, 0.98, nt))
            c = coherence.incoherence(fam, ps)
            d = dutchbook.dutch_book(fam, ps)
            rows.append((fam["family_id"], "random", c["L2"], d["g"]))
            proj_says_coherent = c["L2"] <= TOL_COHERENT
            lp_says_coherent = not d["arbitrage"]
            if proj_says_coherent != lp_says_coherent:
                bad.append(("VERDICTS DISAGREE", fam["family_id"], ps, c["L2"], d["g"]))

    n_coh = sum(1 for r in rows if r[1] == "coherent")
    n_rnd = sum(1 for r in rows if r[1] == "random")
    arb = sum(1 for r in rows if r[1] == "random" and r[3] > TOL_ARB)
    print(f"crosscheck: {len(rows)} trials over {len(families())} family structures")
    print(f"  coherent-by-construction : {n_coh}  (both auditors must find nothing)")
    print(f"  random quotes            : {n_rnd}  ({arb} arbitrageable = {arb / max(n_rnd,1):.0%})")

    # on the random arm the two magnitudes should also co-move, not just agree in sign
    xs = [r[2] for r in rows if r[1] == "random"]
    ys = [r[3] for r in rows if r[1] == "random"]
    if len(xs) > 3 and np.std(xs) > 0 and np.std(ys) > 0:
        print(f"  corr(projection L2, arbitrage g) = {np.corrcoef(xs, ys)[0, 1]:.3f}")

    if bad:
        print(f"\n{len(bad)} PROBLEM(S):")
        for kind, fid, ps, l2, g in bad[:12]:
            print(f"  {kind}  {fid}  L2={l2:.2e} g={g:.2e}")
            print(f"    prices={[round(p, 4) for p in ps]}")
        sys.exit(1)
    print("\nPASS  the projection auditor and the LP auditor agree on every trial")


if __name__ == "__main__":
    main()
