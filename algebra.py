"""Shared event algebra. EVERY module and question file depends on this contract.

A *family* is a small set of base events plus logical constraints. Its state space
is the set of *atoms*: truth assignments over the base events that the constraints
permit. Every elicited quantity (a *target*) is either

    unconditional:  P(expr)
    conditional:    P(expr | given)

where `expr`/`given` are boolean formulas over base-event names using
`and`, `or`, `not`, and parentheses.

Because the state space is explicit and finite, coherence has an exact meaning:
a set of judgments is coherent iff some distribution q over atoms reproduces all
of them. Everything else in this repo (QP projection, Dutch-book LP, scoring) is
built on the two matrices this module emits.

de Finetti bet payoffs, per atom w, for a stake of 1 unit at quoted price p:

    unconditional on E:      1[w in E] - p
    conditional on E given G: (1[w in E] - p) if w in G else 0      (called-off bet)
"""
import re

_ALLOWED_TOKEN = re.compile(r"^(and|or|not|\(|\)|[A-Za-z_][A-Za-z0-9_]*)$")


class FamilyError(ValueError):
    """Raised when a family dict violates the schema or its own constraints."""


def _tokens(expr):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\(|\)", expr)


def check_expr(expr, names):
    """Validate a boolean formula against the family's base-event names."""
    if not isinstance(expr, str) or not expr.strip():
        raise FamilyError(f"empty expression: {expr!r}")
    for tok in _tokens(expr):
        if not _ALLOWED_TOKEN.match(tok):
            raise FamilyError(f"illegal token {tok!r} in {expr!r}")
        if tok not in ("and", "or", "not", "(", ")") and tok not in names:
            raise FamilyError(f"unknown base event {tok!r} in {expr!r} (have {sorted(names)})")
    return True


def evaluate(expr, atom):
    """Truth value of `expr` in `atom` (a dict base_event_name -> bool)."""
    return bool(eval(expr, {"__builtins__": {}}, dict(atom)))  # noqa: S307 - tokens whitelisted


def _satisfies(atom, constraints):
    for c in constraints:
        t = c["type"]
        if t == "exclusive":
            if sum(atom[e] for e in c["events"]) > 1:
                return False
        elif t == "exhaustive":
            if not any(atom[e] for e in c["events"]):
                return False
        elif t == "implies":
            if atom[c["a"]] and not atom[c["b"]]:
                return False
        else:
            raise FamilyError(f"unknown constraint type {t!r}")
    return True


def atoms(family):
    """All constraint-permitted truth assignments over the base events.

    Returned in a deterministic order (base-event names sorted; False before True),
    so atom indices are stable across processes and runs.
    """
    names = sorted(family["base_events"])
    out = []
    for mask in range(2 ** len(names)):
        atom = {n: bool(mask >> i & 1) for i, n in enumerate(names)}
        if _satisfies(atom, family.get("constraints", [])):
            out.append(atom)
    if not out:
        raise FamilyError(f"{family['family_id']}: constraints admit no atoms")
    return out


def validate(family):
    """Full structural check. Returns the atom list. Raises FamilyError."""
    for k in ("family_id", "domain", "base_events", "targets"):
        if k not in family:
            raise FamilyError(f"family missing {k!r}")
    names = set(family["base_events"])
    if len(names) < 2:
        raise FamilyError(f"{family['family_id']}: need >=2 base events")
    for n, ev in family["base_events"].items():
        for k in ("text", "resolution", "source", "resolve_by"):
            if not ev.get(k):
                raise FamilyError(f"{family['family_id']}.{n} missing {k!r}")
    for c in family.get("constraints", []):
        for e in (c.get("events") or [c.get("a"), c.get("b")]):
            if e not in names:
                raise FamilyError(f"{family['family_id']}: constraint names {e!r}")
    seen = set()
    A = atoms(family)
    for t in family["targets"]:
        for k in ("tid", "kind", "text", "expr"):
            if not t.get(k):
                raise FamilyError(f"{family['family_id']}: target missing {k!r}: {t}")
        if t["tid"] in seen:
            raise FamilyError(f"duplicate tid {t['tid']!r}")
        seen.add(t["tid"])
        check_expr(t["expr"], names)
        if t["kind"] == "conditional":
            check_expr(t.get("given", ""), names)
            if not any(evaluate(t["given"], a) for a in A):
                raise FamilyError(f"{t['tid']}: conditioning event is impossible under constraints")
        elif t["kind"] != "unconditional":
            raise FamilyError(f"{t['tid']}: kind must be unconditional|conditional")
    if len(family["targets"]) < 3:
        raise FamilyError(f"{family['family_id']}: need >=3 targets to test any axiom")
    return A


def target_masks(family):
    """(atoms, list of (event_mask, given_mask)) as lists of bools over atoms.

    given_mask is all-True for unconditional targets.
    """
    A = atoms(family)
    masks = []
    for t in family["targets"]:
        ev = [evaluate(t["expr"], a) for a in A]
        gv = [evaluate(t["given"], a) for a in A] if t["kind"] == "conditional" else [True] * len(A)
        masks.append((ev, gv))
    return A, masks


def implied(q, ev, gv):
    """Probability of a target under atom distribution q. None if conditioned on a null set."""
    denom = sum(qi for qi, g in zip(q, gv) if g)
    if denom <= 1e-12:
        return None
    return sum(qi for qi, e, g in zip(q, ev, gv) if e and g) / denom


def payoff_matrix(family, prices):
    """P[i][j] = payoff on atom i of a 1-unit bet on target j at the quoted price.

    `prices` is a list aligned with family["targets"].
    """
    A, masks = target_masks(family)
    P = []
    for i in range(len(A)):
        row = []
        for j, (ev, gv) in enumerate(masks):
            row.append((1.0 if ev[i] else 0.0) - prices[j] if gv[i] else 0.0)
        P.append(row)
    return P


def load_all(qdir="questions"):
    """Read every questions/*.json, validate, return a flat list of families."""
    import json
    import pathlib

    fams, errs = [], []
    for p in sorted(pathlib.Path(qdir).glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            errs.append(f"{p.name}: unreadable ({e})")
            continue
        for fam in (data if isinstance(data, list) else data.get("families", [])):
            try:
                validate(fam)
                fam["_file"] = p.name
                fams.append(fam)
            except FamilyError as e:
                errs.append(f"{p.name}: {e}")
    ids = [f["family_id"] for f in fams]
    for fid in set(ids):
        if ids.count(fid) > 1:
            errs.append(f"duplicate family_id {fid!r}")
    return fams, errs


if __name__ == "__main__":
    fams, errs = load_all()
    print(f"{len(fams)} families, {sum(len(f['targets']) for f in fams)} targets")
    for e in errs:
        print("  ERR", e)
