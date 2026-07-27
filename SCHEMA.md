# Contract

Everything in this repo agrees on `algebra.py`. Read it before writing code.
Python 3.12, numpy 1.26, scipy 1.13. **No cvxpy, no pandas, no sklearn** — stdlib + numpy + scipy only.

## Question file: `questions/<domain>.json`

A JSON **list** of *family* objects.

```json
[
  {
    "family_id": "geo_hormuz_closure",
    "domain": "geopolitics",
    "base_events": {
      "A": {
        "text": "Iran formally closes the Strait of Hormuz to commercial traffic for >=72 consecutive hours",
        "resolution": "Resolves YES if Lloyd's List Intelligence or the US 5th Fleet reports commercial transit suspended for 72+ consecutive hours",
        "source": "Lloyd's List Intelligence daily transit report",
        "resolve_by": "2027-03-31"
      },
      "B": { "text": "...", "resolution": "...", "source": "...", "resolve_by": "2027-03-31" }
    },
    "constraints": [ {"type": "implies", "a": "A", "b": "B"} ],
    "targets": [
      {"tid": "geo_hormuz_closure.A",    "kind": "unconditional", "text": "What is the probability that <A text>?", "expr": "A"},
      {"tid": "geo_hormuz_closure.B",    "kind": "unconditional", "text": "...", "expr": "B"},
      {"tid": "geo_hormuz_closure.AandB","kind": "unconditional", "text": "...both...", "expr": "A and B"},
      {"tid": "geo_hormuz_closure.A_g_B","kind": "conditional",   "text": "Assume <B text> occurs. What is the probability that <A text>?", "expr": "A", "given": "B"}
    ],
    "axioms_tested": ["product_rule", "conjunction_bound", "monotonicity"]
  }
]
```

Rules:
- `constraints`: `{"type":"exclusive","events":[...]}`, `{"type":"exhaustive","events":[...]}`, `{"type":"implies","a":X,"b":Y}`. Omit if none. Only declare a constraint that is **logically** forced, never one that is merely likely.
- `expr` / `given`: boolean formulas over base-event names only — `and`, `or`, `not`, parens.
- `tid` must be globally unique; prefix with `family_id`.
- `text` is the exact question shown to the forecaster. It must be self-contained: a fresh model with no other context must be able to answer it. Spell out the event, don't refer to "A".
- **Every event must be genuinely unresolved as of 2026-07-26** and resolve between 2026-09-01 and 2027-06-30. This is the whole point: no hindsight. Use WebSearch to confirm the event has not already happened and that the named resolution source exists.
- Resolution criteria must be crisp enough that two strangers agree, and name a specific public source.

## Module contracts

Import `algebra` and use `atoms`, `target_masks`, `implied`, `payoff_matrix`, `validate`.

A **forecast record** (what `elicit.py` writes, what the audits read):

```json
{"tid": "...", "model": "claude-sonnet-5", "p": 0.34, "rationale": "one sentence", "asked_utc": "..."}
```

`out/forecasts_<model>.json` is a list of those, one per target.

Each module exposes `run()` writing its own `out/*.json`, and is runnable as
`python3 <module>.py` for a self-test on a hand-checkable toy family. Print a short
summary to stdout. Never write outside your assigned file(s).
