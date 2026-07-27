# forecast-integrity

**Auditing a probabilistic forecaster before any of its forecasts resolve.**

A forecasting product that answers questions about 2027 cannot be evaluated on accuracy until 2027.
That is an uncomfortable position to sell from, and it is the position every long-horizon forecasting
system is in. This repo scores one anyway, using three properties that are checkable on day zero:

1. **Coherence** — do its probabilities obey the probability axioms?
2. **Arbitrage** — can a book be built against its own quoted prices that wins on every outcome?
3. **Contamination** — is its apparent skill forecasting, or recall of events it already knows?

None of the three needs a single resolved question. All three are computed here on 40 question
families whose events resolve between 2026-09-01 and 2027-06-30, each web-verified as unresolved at
authoring time, with all 1,920 forecasts sealed in Merkle commitments before analysis touched them.

---

## Results

Two models × two elicitation regimes, 40 families / 240 targets per arm.

| | Sonnet 5 fresh | Opus 5 fresh | Sonnet 5 joint | Opus 5 joint |
|---|---|---|---|---|
| Families with an arbitrage | 40/40 | 40/40 | 39/40 | 35/40 |
| **Mean guaranteed loss per $100 staked** | **$4.45** | **$3.16** | **$0.42** | **$0.05** |
| Worst family | $15.00 | $8.64 | $2.50 | $0.36 |
| Distance from coherent set (mean L2) | 0.0440 | 0.0334 | 0.0052 | 0.0011 |
| Axiom violations | 79 | 79 | 57 | 38 |
| **× the sampling-noise floor** | **4.91×** | **2.63×** | 0.49× | 0.04× |
| p(noise explains it) | 0.000 | 0.005 | 0.762 | 0.985 |

*Fresh* = each question asked in a separate process with no memory of the others. *Joint* = all of a
family's questions asked in one prompt, so the model can see its own numbers.

### 1. Asked independently, the forecaster books a guaranteed loss

Both models are arbitrageable on **every single family**. The worst book is not numerical drift —
Sonnet quoted:

```
P(Alphabet 2027 capex guidance ≥ $260B)          = 0.35
P(Meta 2027 capex guidance ≥ $185B)              = 0.40
P(both)                                          = 0.05
P(either)                                        = 0.10   ← impossible
```

A disjunction cannot be less likely than either disjunct. Selling the two singles and buying the
conjunction and disjunction extracts **$15 per $100 staked on every one of the 6 possible
outcomes**. No amount of waiting for 2027 finds this faster than the audit does.

### 2. It is not sampling noise

The decisive control, and the one that nearly killed the result. Each target was re-asked 3 times
(480 extra calls per model) to measure test–retest SD: **0.0175 for Sonnet, 0.0250 for Opus**. Every
observed number is then bracketed between

- a **ceiling** — text-blind random quotes, which arbitrage at ~$17.5 per $100 and are arbitrageable
  ~100% of the time (so "fraction of families arbitrageable" is a worthless statistic and is not
  reported as a finding), and
- a **floor** — the model's *own repaired forecast*, re-perturbed at its own measured retest SD and
  re-audited. That is the arbitrage a perfectly coherent forecaster would show purely from failing to
  repeat itself.

Fresh-arm arbitrage lands at 4.91× (Sonnet) and 2.63× (Opus) that floor, at p = 0.000 and p = 0.005.
Note that the naive scalar comparison — mean retest SD vs mean axiom gap — reads 0.51 and **0.96**,
and for Opus would have said "not separable." Propagating the noise through the actual LP shows why
that is wrong: independent jitter partially cancels inside a book, systematic axiom violations do
not. The crude ratio would have thrown away a real effect.

### 3. Joint context fixes the bookkeeping, not the arithmetic

Letting the model see its own numbers collapses the arbitrage by 11× (Sonnet) and 63× (Opus), to at
or below the noise floor. Note that an arbitrage still *exists* on 39/40 and 35/40 families — it is
just worth pennies. This is the clearest illustration of why magnitude is the metric and the count
is not: on the count alone, joint elicitation looks like it changed almost nothing.

*Which* violations disappear is the interesting part:

| axiom | Opus fresh | Opus joint |
|---|---|---|
| complement (sums to 1) | 9 | **0** |
| partition (bands sum to 1) | 3 | **0** |
| monotonicity (nested events) | 1 | **0** |
| conjunction bound | 6 | **0** |
| disjunction identity | 11 | **0** |
| **product rule** P(A∧B) = P(A\|B)P(B) | 36 | **27** |
| **law of total probability** | 13 | **11** |

The additive axioms are pure bookkeeping: they vanish entirely once the model can see its own
numbers. The **multiplicative** ones survive — those are arithmetic failures, not bookkeeping
failures, and no amount of context fixes them.

That is the actionable finding for anyone building this. Any system that elicits probabilities from a
question queue — i.e. independently — carries a $3–4 per $100 structural loss that is free to remove,
either by joint elicitation or by applying the projection repair in `coherence.py`. The Bayes errors
that remain need something else.

### 4. Contamination: recall is not skill

| | resolved controls (2024–25) | open questions (this set) |
|---|---|---|
| n | 18 (9 YES / 9 NO) | 178 |
| Brier | 0.051 | — unresolved |
| Skill vs uninformative | +0.796 | — |
| Accuracy | 0.944 | — |
| Mean confidence \|p−0.5\| | 0.469 | 0.300 |

Sonnet is near-perfectly sharp on events it has already read about (0.469 out of a maximum possible
0.50) and materially vaguer on open ones. The **confidence gap is +0.170, implying that 36% of the
sharpness a backtest would show does not transfer to a live forecast** — the discount a buyer should
apply to any backtest-based claim.

The date probe found **0 of 40 families that the model reads as already resolved**, and dated 100% of
them to within two months of the true authoring month. The question set is certified hindsight-free,
which is what makes the gap above interpretable rather than circular.

---

## Why the design is the way it is

Three ways this could have been fooled, and what stops each:

**"You asked them separately on purpose."** Yes — a model that sees its own prior answers can enforce
consistency without holding coherent beliefs, which measures prompt engineering rather than the
forecaster. Both arms are run, so the difference is measured (§3) rather than assumed.

**"The incoherence is just noise."** §2. The floor is measured, not asserted, and the repo prints a
refusal — `!! NOT clearly above sampling noise — do not headline this number` — whenever an arm fails
it. Two arms do fail it, and are reported as failing it.

**"Your two auditors are the same computation twice."** They aren't. `coherence.py` projects onto the
coherent set with a nonconvex QP; `dutchbook.py` solves an LP for a guaranteed-profit book. They
share no code beyond the event algebra, and de Finetti's theorem says they must agree on the yes/no
question. `crosscheck.py` tests exactly that over 300 trials across 5 family structures: **they never
disagree**, and their magnitudes correlate 0.917. Agreement is a test of both, not a tautology.

---

## Layout

| file | what it does |
|---|---|
| `algebra.py` | the shared contract: atoms of a family, boolean event expressions, de Finetti bet payoffs |
| `questions/*.json` | 40 families / 240 targets — geopolitics & defense, energy & supply chain, macro & rates, semis, catastrophe-insurance & pharma |
| `elicit.py`, `llm.py` | the forecaster under audit — cached headless `claude -p`, one fresh context per target, `--repeat` and `--joint` arms |
| `coherence.py` | projection onto the coherent set, per-axiom violation detection, and the repair |
| `dutchbook.py` | the LP that builds and prices the arbitrage, with a legible certificate |
| `crosscheck.py` | asserts the two auditors agree (de Finetti duality, numerically) |
| `baselines.py` | noise floor and ceiling — the controls that decide if a number is real |
| `leakage.py`, `controls_resolved.json` | date-de-anonymization probe + 18 web-verified resolved controls |
| `ledger.py` | Merkle commitment; `out/ROOTS.txt` is the publishable artefact |
| `scoring.py` | Brier, log score, Murphy decomposition, reliability curve, horizon decay — armed, not yet runnable |
| `report.py` | the HTML report |

Python 3.12, numpy, scipy. No other dependencies.

```bash
python3 run_all.py                          # validate → elicit → commit → audit → score → report
python3 crosscheck.py                       # the duality test, no model calls
python3 elicit.py --repeat 3 --dry-run      # always count the bill before a sweep
```

Every module self-tests against hand-computed values. The complement case (`P(A)=0.6`, `P(¬A)=0.6`)
must give projection L2 = 0.1 and arbitrage g = 0.2; both hold to 1e-6, derived independently.
Total cost of every model call behind every number above: **$21.63** across 1,590 cached calls
(summed from `cache/*.json`). Re-running the full pipeline bills $0.00 — the cache is keyed on
`sha256(model, variant, prompt)`, so nothing is ever paid for twice.

### Sealed commitments

```
ledger_claude-sonnet-5.json         n=720   f92de5c361e231fb0ff0ec0f6817bb703151cfd2257ba4078f46bc249935a6b4
ledger_claude-opus-5.json           n=720   36a72edbd9d95e3f1191adc256d8422cbb11584ada6ea906b6612963684a2f78
ledger_claude-sonnet-5_joint.json   n=240   e67c8b05c3052e29203b9c09e80b295e85d322e529d18499280d08e982aa9431
ledger_claude-opus-5_joint.json     n=240   c71354f180934469e787bf36475237597f18cd221c9efc81879b29b5385d9cb4
```

Sealed 2026-07-27. Earliest resolution **2026-11-15** (`en_eu_gas_storage_winter.EU_S70`).
An unpublished root is not a commitment — publishing `out/ROOTS.txt` is the point.

---

## What this does not show

- **Coherence is necessary, not sufficient.** A forecaster answering 0.5 to everything is perfectly
  coherent and perfectly useless. This is a filter, not a ranking.
- **The contamination gap is an upper bound**, not a point estimate. The resolved controls are famous
  events and the open questions are deliberately near-even, so part of the gap is the past genuinely
  being more knowable. The clean version is a matched-difficulty pre/post-cutoff split, which becomes
  runnable when this set resolves.
- **The date probe partly measures where the knowledge cutoff sits**, not only leakage.
- **The audited book is draw 0** of each target. Averaging all 3 draws moves Sonnet's fresh arm from
  $4.45 to $4.36 per $100 — the headline does not depend on which draw was used, but the two numbers
  appear in different files for that reason.
- **Accuracy, calibration and horizon decay are unmeasured here, by construction.** That is what the
  commitment is for: the forecasts are sealed now, so those become measurable later on numbers that
  could not have been edited in hindsight. `scoring.py` turns on when outcomes land in
  `out/resolutions.json`.

This repo measures what can be measured today and refuses to fabricate the rest.
