# Can an LLM be made coherent in-context?

Short answer: **yes, but only implicitly. Told explicitly what it got wrong, it gets worse.**

RLCR puts a calibration term in the RL reward. This asks whether the same pressure can be
applied purely in the context window, with no gradient — and scores the answer against
events that have already resolved, so the accuracy numbers are real rather than promissory.

## Setup

10 question families whose events resolved in 2024–25 (Moldova/EU accession, German
government formation, Fed target range, BoJ path, WTI/Hormuz, US nuclear restarts, Nvidia
scale, Intel rescue, US property-cat losses, obesity-drug approvals). Each family is 2–3
entangled base events and 4–6 questions about them — singles, conjunctions, disjunctions,
conditionals — chosen so the probability axioms are testable.

58 of 60 targets have a defined outcome (2 are called-off conditionals: `P(A|B)` where B
did not occur, which de Finetti leaves undefined; they still constrain the repair).
Base rate 0.50, so a forecaster that always says 0.5 scores Brier 0.2500.

Four arms, same model (Claude Sonnet 5), same questions:

| arm | what it is |
|---|---|
| **A fresh** | each question in a separate process, no memory of the others |
| **B joint** | all of a family's questions in one prompt — implicit coherence pressure |
| **C feedback** | A's answers + the named axiom violations + the dollar cost, then "revise" |
| **D repair** | A's answers projected onto the coherent set by least squares — no model involved |

## Result

```
arm                      Brier    skill    $/100 arbitrage    mean |p-0.5|
A fresh                 0.2203    11.9%        9.82              0.413
B joint                 0.1558    37.7%        0.32              0.371
C feedback              0.2778   -11.1%       11.66              0.392
D repair (no model)     0.1917    23.3%        0.00              0.361
```

**B — implicit pressure works.** Simply putting the six questions in one context triples
skill (11.9% → 37.7%) and removes almost all the arbitrage. Nothing was explained to the
model; it just saw its own answers side by side.

**C — explicit pressure fails, and actively harms.** Shown its own numbers, the specific
axioms they violate, and the fact that $9.82 per $100 can be extracted from them, the model
revises into a set that is *less* accurate than where it started (skill 11.9% → **−11.1%**,
worse than always answering 0.5) and *less* coherent ($9.82 → **$11.66**). It was told
exactly what was wrong and made it worse.

**D — the mechanical repair works, and needs no model.** Least-squares projection onto the
coherent set doubles skill (11.9% → 23.3%) and by construction removes all arbitrage.

Confounds checked: the repair does reduce overconfidence (mean |p−0.5| 0.413 → 0.361), and
since the base rate is exactly 0.50, shrinking toward 0.5 improves Brier mechanically. A
matched-shrinkage control — same confidence reduction, no coherence structure — reaches only
0.2043, so roughly 56% of D's gain is de-biasing and 44% is genuine structure. An
oracle-tuned shrinkage reaches 0.1892, marginally better than D, but requires the answers
you are trying to predict in order to pick its shrinkage factor. D requires nothing.
Sign test on D: repair improves 10 of 10 families, one-sided p = 0.001.

## Reading

The model can hold a coherent set of beliefs when the questions are presented together. It
cannot *reason its way* to one when the inconsistency is pointed out. Those are different
capabilities, and only the first is available in current models.

This lines up with the larger run in this repo (40 unresolved families, 240 targets, two
models). There, joint context zeroed every additive violation — complement, partition,
monotonicity — while the multiplicative ones barely moved: Sonnet's law-of-total-probability
count went 13 → 13, unchanged. Seeing its own numbers lets it check that things sum to one.
It does not confer the ability to do arithmetic on conditional probabilities, and asking it
to do so explicitly (arm C) is worse than not asking.

## Caveats

- 10 families, 58 scored targets, one model. This is a pilot, not an established effect.
- These events resolved in 2024–25, so recall inflates every absolute Brier here. All four
  arms use the identical questions, so the *comparison* is controlled; the absolute skill
  numbers are not transferable to genuinely open questions.
- Arm C was run once per family with one prompt design. A different framing might do better,
  and "explicit feedback fails" should be read as "this natural framing fails," not as a
  proof about all framings.
- Arm C's revisions are matched back to targets **by position**, because the model invents
  its own identifiers when not shown them. An earlier version of this script silently fell
  back to the unrevised numbers; that bug is fixed and the table above is post-fix.

## Reproduce

```bash
python3 repair_test.py --model claude-sonnet-5 --joint   # arms A, B, D
python3 selfrepair.py                                    # all four arms
```

Everything is cached; a re-run bills $0.00.
