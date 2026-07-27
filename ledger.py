"""Tamper-evident commitment ledger for forecasts.

You cannot claim a forecast was a *forecast* unless it existed before the outcome
was known. This module is the mechanism for that claim: it reduces a whole set of
elicited probabilities to one 64-character string (a Merkle root) that can be
published cheaply and irrevocably (gist, git tag, tweet, OpenTimestamps) *before*
any question resolves. Afterwards, anyone holding the forecast records can
recompute the root; a single flipped digit anywhere changes it.

What is bound by the hash
-------------------------
`canonical()` hashes exactly the fields that constitute the claim:

    CLAIM_FIELDS = ("tid", "model", "p")

Nothing else. Rewriting a rationale, adding a `notes` field, or reordering keys
leaves the root untouched -- cosmetic edits are explicitly permitted. Changing a
probability, a target id, or the attributed model breaks it.

`asked_utc` is deliberately *not* hashed. A self-reported timestamp inside the
record proves nothing (the forecaster writes it). The trusted timestamp is the
publication time of the root, which lives outside this repo. Hashing a
self-asserted clock would create the illusion of a guarantee it cannot provide.

Merkle convention (fixed, and repeated in `merkle_root`'s docstring)
-------------------------------------------------------------------
* hash function: SHA-256 throughout.
* leaf:   ``sha256(canonical(record))``, hex.
* node:   ``sha256(left_digest_bytes || right_digest_bytes)`` -- the *raw 32-byte*
          digests are concatenated, not their hex spellings.
* odd level: **duplicate the last node** of that level and pair it with itself
          (the Bitcoin convention). Applied at every level independently.
* one leaf: the root *is* that leaf; no extra hashing.
* zero leaves: the root is ``sha256(b"")`` (``EMPTY_ROOT``), a sentinel.

The duplicate-last-node rule is known to make tree *shape* ambiguous (an
attacker can sometimes find a different leaf multiset with the same root by
replaying duplicated tails). `commit()` neutralizes that by recording ``n`` and
the full ``leaf_hashes`` list next to the root, and `verify()` checks the count
before it checks the root, so a shape-collision is rejected on arity alone.

Public API
----------
    canonical(record) -> bytes
    leaf_hash(record) -> str (hex)
    merkle_root(leaves) -> str (hex)
    merkle_proof(leaves, i) -> [{"hash": hex, "side": "left"|"right"}, ...]
    verify_proof(leaf, proof, root) -> bool
    commit(records, asof_utc, meta=None) -> dict
    verify(ledger_path, records) -> dict
    run(forecasts_glob="out/forecasts_*.json", asof_utc=None) -> [dict, ...]

No network access, ever. Publishing the root is a human step; this module only
writes ``out/ledger_<model>.json`` and a plain-text ``out/ROOTS.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import pathlib
import re
import sys
from datetime import datetime, timezone

ROOT_DIR = pathlib.Path(__file__).resolve().parent

CLAIM_FIELDS = ("tid", "model", "p")
HASH_ALG = "sha256"
EMPTY_ROOT = hashlib.sha256(b"").hexdigest()
ODD_NODE_CONVENTION = "duplicate-last-node"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class LedgerError(ValueError):
    """Raised when a record cannot be canonicalized or a tree input is malformed."""


# --------------------------------------------------------------------------- #
# canonicalization + leaf hashing
# --------------------------------------------------------------------------- #
def canonical(record):
    """Deterministic serialization of the *claim* inside a forecast record.

    Returns ``json.dumps`` bytes over exactly ``CLAIM_FIELDS`` with sorted keys
    and no whitespace, i.e. ``{"model":...,"p":...,"tid":...}``. Fields outside
    that set (``rationale``, ``asked_utc``, anything else) are dropped, so
    cosmetic edits cannot move the hash.

    Type coercion is part of the canonical form: ``tid``/``model`` are stringified
    and ``p`` is coerced to ``float`` (so ``0`` and ``0.0`` and ``"0.0"`` all hash
    alike). ``p`` must be a finite probability in [0, 1]. ``float.__repr__`` is
    the shortest round-tripping decimal for an IEEE-754 double, so the byte
    string is stable across machines and Python builds.
    """
    if not isinstance(record, dict):
        raise LedgerError(f"record must be a dict, got {type(record).__name__}")
    claim = {}
    for k in CLAIM_FIELDS:
        if k not in record or record[k] is None:
            raise LedgerError(f"record missing claim field {k!r}: {record!r}")
    for k in ("tid", "model"):
        v = record[k]
        if not isinstance(v, (str, int, float)):
            raise LedgerError(f"{k!r} must be a scalar, got {type(v).__name__}")
        s = str(v).strip()
        if not s:
            raise LedgerError(f"{k!r} is empty in {record!r}")
        claim[k] = s
    try:
        p = float(record["p"])
    except (TypeError, ValueError) as e:
        raise LedgerError(f"p is not a number: {record['p']!r}") from e
    if not math.isfinite(p) or not (0.0 <= p <= 1.0):
        raise LedgerError(f"p must be a finite probability in [0,1], got {p!r}")
    claim["p"] = p
    return json.dumps(
        claim, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def leaf_hash(record):
    """Hex SHA-256 of ``canonical(record)``."""
    return hashlib.sha256(canonical(record)).hexdigest()


def leaf_hashes(records):
    """Leaf hashes for a list of records, in the order given (order is committed)."""
    return [leaf_hash(r) for r in records]


def committable(records):
    """Split records into those that carry a claim and those that do not.

    `elicit.py` writes ``p: None`` plus an ``error`` for a call it could not parse.
    That row is not a forecast, and hashing it would be committing to the absence
    of a claim. But refusing to commit the *whole file* because one API call
    failed is strictly worse: it leaves everything uncommitted, and "uncommitted"
    is the one state that cannot be repaired once an outcome is known.

    So null/malformed rows are excluded **by index, on the record**, and the rest
    are committed. `verify()` re-derives the same split and cross-checks it
    against the ledger's declared ``excluded_indices``, so blanking a real ``p``
    to ``None`` after the fact is detected rather than absorbed.

    Returns ``(keep, excluded)`` where excluded entries are
    ``{"index","tid","reason"}``.
    """
    keep, excluded = [], []
    for i, r in enumerate(records):
        reason = None
        if not isinstance(r, dict):
            reason = f"not a dict ({type(r).__name__})"
        elif r.get("p") is None:
            reason = "p is null" + (f": {str(r.get('error'))[:80]}" if r.get("error") else "")
        else:
            try:
                canonical(r)
            except LedgerError as e:
                reason = str(e)
        if reason is None:
            keep.append(r)
        else:
            tid = r.get("tid") if isinstance(r, dict) else None
            excluded.append({"index": i, "tid": None if tid is None else str(tid), "reason": reason})
    return keep, excluded


# --------------------------------------------------------------------------- #
# Merkle tree
# --------------------------------------------------------------------------- #
def _hex(h, what="hash"):
    if not isinstance(h, str) or not _HEX64.match(h.lower()):
        raise LedgerError(f"{what} is not a 64-char lowercase hex sha256: {h!r}")
    return h.lower()


def _node(left_hex, right_hex):
    """Internal node: sha256 over the two *raw* 32-byte digests, concatenated."""
    return hashlib.sha256(
        bytes.fromhex(_hex(left_hex, "left")) + bytes.fromhex(_hex(right_hex, "right"))
    ).hexdigest()


def merkle_root(leaves):
    """Merkle root (hex) of a list of hex leaf hashes.

    Convention, stated explicitly because it must be reproducible by a stranger
    with nothing but this docstring:

    1. SHA-256 everywhere.
    2. A level is reduced pairwise, left-to-right:
       ``parent = sha256(raw32(left) || raw32(right))`` -- the raw binary
       digests are concatenated, *not* their hex strings.
    3. **Odd level -> duplicate the last node** and pair it with itself. This is
       re-applied independently at every level that happens to be odd.
    4. A single leaf is its own root (no further hashing).
    5. An empty leaf list has root ``EMPTY_ROOT = sha256(b"")``.

    Worked example for three leaves l0, l1, l2 (the odd case)::

        root = N( N(l0, l1), N(l2, l2) )

    Item 3 makes tree shape ambiguous in isolation; `commit()` therefore stores
    ``n`` and every leaf hash beside the root, and `verify()` compares counts
    first, which is what closes the hole.
    """
    if not leaves:
        return EMPTY_ROOT
    level = [_hex(h, f"leaf[{i}]") for i, h in enumerate(leaves)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = level + [level[-1]]  # odd node: duplicate the last
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves, i):
    """Inclusion proof for ``leaves[i]``: a list of ``{"hash","side"}`` steps.

    ``side`` is the position of the *sibling* relative to the running hash, so a
    verifier replays it as::

        side == "left"  ->  h = node(sibling, h)
        side == "right" ->  h = node(h, sibling)

    Steps run leaf-ward to root-ward. When the odd-node rule duplicates the last
    node, the sibling of that node is itself, which shows up as a step whose
    ``hash`` equals the running hash with ``side == "right"``.
    """
    if not leaves:
        raise LedgerError("no leaves to prove")
    n = len(leaves)
    if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < n):
        raise LedgerError(f"leaf index {i!r} out of range for {n} leaves")
    level = [_hex(h, f"leaf[{k}]") for k, h in enumerate(leaves)]
    idx = i
    proof = []
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = level + [level[-1]]
        sib = idx ^ 1
        proof.append({"hash": level[sib], "side": "left" if sib < idx else "right"})
        level = [_node(level[k], level[k + 1]) for k in range(0, len(level), 2)]
        idx //= 2
    return proof


def verify_proof(leaf, proof, root):
    """True iff replaying ``proof`` from ``leaf`` reproduces ``root``.

    Adversarial input is expected here, so malformed hashes/steps return False
    rather than raising. The final comparison uses `hmac.compare_digest`.
    """
    try:
        h = _hex(leaf, "leaf")
        target = _hex(root, "root")
        for step in proof:
            side = step["side"]
            sib = _hex(step["hash"], "proof step")
            if side == "left":
                h = _node(sib, h)
            elif side == "right":
                h = _node(h, sib)
            else:
                return False
    except (LedgerError, KeyError, TypeError, AttributeError):
        return False
    return hmac.compare_digest(h, target)


# --------------------------------------------------------------------------- #
# commit / verify
# --------------------------------------------------------------------------- #
def commit(records, asof_utc, meta=None):
    """Reduce ``records`` to a publishable commitment dict.

    Keys: ``root``, ``asof_utc``, ``n``, ``models`` (sorted distinct) plus
    ``model`` when there is exactly one, ``leaf_hashes`` and ``tids`` (both in
    committed order), and the self-describing ``hash_alg`` /
    ``canonical_fields`` / ``odd_node_convention`` / ``empty_root`` fields so the
    file can be checked without this source.

    ``meta`` is merged in first; the core keys above always win, so metadata can
    never quietly redefine the commitment.
    """
    if not isinstance(records, list):
        raise LedgerError("records must be a list")
    if not isinstance(asof_utc, str) or not asof_utc.strip():
        raise LedgerError("asof_utc must be a non-empty ISO-8601 string")
    leaves = leaf_hashes(records)
    tids = [str(r["tid"]) for r in records]
    models = sorted({str(r["model"]) for r in records})
    out = dict(meta or {})
    out.update(
        {
            "root": merkle_root(leaves),
            "asof_utc": asof_utc,
            "n": len(records),
            "models": models,
            "leaf_hashes": leaves,
            "tids": tids,
            "hash_alg": HASH_ALG,
            "canonical_fields": list(CLAIM_FIELDS),
            "odd_node_convention": ODD_NODE_CONVENTION,
            "empty_root": EMPTY_ROOT,
        }
    )
    if len(models) == 1:
        out["model"] = models[0]
    dupes = sorted({t for t in tids if tids.count(t) > 1})
    if dupes:
        out["duplicate_tids"] = dupes[:20]
    return out


def verify(ledger_path, records):
    """Recompute the root from ``records`` and compare with a stored ledger.

    Returns a dict::

        {"ok", "match", "ledger_path", "root_ledger", "root_recomputed",
         "n_ledger", "n_records", "mismatched_leaves", "notes", ...}

    ``mismatched_leaves`` names the offending leaf *indices* (with tids and both
    hashes) so tampering is localized, not just detected. Count is compared
    before the root, which is what makes the duplicate-last-node convention safe.

    Rows with ``p: None`` are excluded exactly as `run()` excluded them (see
    `committable`), and the exclusion set itself is compared against the ledger's
    record of it. ``ok`` is True only when the root matches, every leaf matches,
    the counts agree, and the ledger declares this module's own conventions;
    ``notes`` explains any failure. Non-fatal remarks are also in ``notes`` but
    do not clear ``ok`` on their own.
    """
    path = _resolve(ledger_path)
    hard = []  # findings that must clear `ok`
    res = {
        "ok": False,
        "match": False,
        "ledger_path": str(path),
        "n_records": len(records) if isinstance(records, list) else None,
        "mismatched_leaves": [],
        "notes": [],
    }

    def fatal(msg):
        hard.append(msg)
        res["notes"].append(msg)
    try:
        led = json.loads(path.read_text())
    except FileNotFoundError:
        res["notes"].append(f"ledger not found: {path}")
        return res
    except Exception as e:  # noqa: BLE001 - a corrupt ledger is a verification failure
        res["notes"].append(f"ledger unreadable: {e}")
        return res

    res["asof_utc"] = led.get("asof_utc")
    res["root_ledger"] = led.get("root")
    res["n_ledger"] = led.get("n")
    led_leaves = led.get("leaf_hashes") or []
    led_tids = led.get("tids") or [None] * len(led_leaves)

    if led.get("hash_alg", HASH_ALG) != HASH_ALG:
        fatal(f"ledger uses hash_alg={led.get('hash_alg')!r}, this code is {HASH_ALG}")
    if led.get("canonical_fields") and list(led["canonical_fields"]) != list(CLAIM_FIELDS):
        fatal(f"ledger canonical_fields={led['canonical_fields']} != {list(CLAIM_FIELDS)}")
    if led.get("odd_node_convention", ODD_NODE_CONVENTION) != ODD_NODE_CONVENTION:
        fatal(
            f"ledger odd_node_convention={led.get('odd_node_convention')!r} != {ODD_NODE_CONVENTION!r}"
        )

    if not isinstance(records, list):
        fatal("records must be a list")
        return res
    keep, excluded = committable(records)
    res["n_committable"] = len(keep)
    res["excluded_indices"] = [e["index"] for e in excluded]
    if excluded:
        res["excluded"] = excluded[:50]
    led_excl = led.get("excluded_indices")
    if led_excl is not None and list(led_excl) != res["excluded_indices"]:
        fatal(
            f"exclusion set differs: ledger excluded {list(led_excl)[:20]}, "
            f"records give {res['excluded_indices'][:20]} "
            "(a claimed p was blanked, or a null was filled in, after the commitment)"
        )
    if led.get("n_source_records") is not None and led["n_source_records"] != len(records):
        fatal(f"source row count differs: ledger saw {led['n_source_records']}, got {len(records)}")
    elif excluded:
        res["notes"].append(f"{len(excluded)} source row(s) carried no claim and were excluded")
    try:
        now_leaves = leaf_hashes(keep)
    except LedgerError as e:
        fatal(f"records cannot be canonicalized: {e}")
        return res
    records = keep
    res["root_recomputed"] = merkle_root(now_leaves)

    if len(now_leaves) != len(led_leaves):
        fatal(f"leaf count differs: ledger has {len(led_leaves)}, records give {len(now_leaves)}")
    for i in range(min(len(now_leaves), len(led_leaves))):
        if not hmac.compare_digest(now_leaves[i].lower(), str(led_leaves[i]).lower()):
            res["mismatched_leaves"].append(
                {
                    "index": i,
                    "tid_ledger": led_tids[i] if i < len(led_tids) else None,
                    "tid_record": str(records[i].get("tid")),
                    "hash_ledger": led_leaves[i],
                    "hash_record": now_leaves[i],
                }
            )
    if len(now_leaves) > len(led_leaves):
        res["extra_records"] = list(range(len(led_leaves), len(now_leaves)))
    elif len(led_leaves) > len(now_leaves):
        res["missing_records"] = list(range(len(now_leaves), len(led_leaves)))

    res["mismatched_indices"] = [m["index"] for m in res["mismatched_leaves"]]
    res["match"] = bool(
        res["root_ledger"]
        and hmac.compare_digest(res["root_recomputed"], str(res["root_ledger"]).lower())
    )
    res["ok"] = bool(
        res["match"]
        and not res["mismatched_leaves"]
        and len(now_leaves) == len(led_leaves)
        and not hard
    )
    if res["ok"]:
        res["notes"].append(f"root matches over {len(now_leaves)} leaves")
    elif not res["match"]:
        res["notes"].append("ROOT MISMATCH - these records are not the committed ones")
    return res


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _resolve(p):
    p = pathlib.Path(p)
    return p if p.is_absolute() else (ROOT_DIR / p)


def _model_label(path, records):
    """Model label for a forecasts file: filename suffix, else the record's model."""
    stem = pathlib.Path(path).stem
    if stem.startswith("forecasts_") and len(stem) > len("forecasts_"):
        return stem[len("forecasts_"):]
    models = sorted({str(r.get("model")) for r in records if r.get("model")})
    return models[0] if len(models) == 1 else "unknown"


def _roots_text(ledgers):
    lines = [
        "# forecast_integrity - forecast commitment roots",
        "#",
        "# Publish this file VERBATIM (gist / git tag / OpenTimestamps / anywhere with a",
        "# third-party timestamp) BEFORE any question resolves. The root is the entire",
        "# claim: without a pre-resolution timestamp on it, a track record is unfalsifiable.",
        "#",
        "# To check a published root against a set of forecast records:",
        "#     python3 ledger.py --verify out/ledger_<model>.json",
        "#",
        f"# hash        : {HASH_ALG}",
        '# leaf        : sha256(json.dumps({"model","p","tid"}, sort_keys=True, separators=(",",":")))',
        "#               (rationale / asked_utc are NOT hashed - cosmetic edits are allowed,",
        "#                and the trusted timestamp is when this file was published, not a",
        "#                timestamp the forecaster wrote inside their own record)",
        "# node        : sha256(raw32(left) || raw32(right))   # raw digests, not hex strings",
        f"# odd node    : {ODD_NODE_CONVENTION} (pair the last node of an odd level with itself)",
        "# single leaf : root == that leaf",
        f"# empty       : root == {EMPTY_ROOT}",
        "#",
    ]
    if not ledgers:
        lines.append("# (no ledgers committed yet)")
    for led in ledgers:
        model = led.get("model") or ",".join(led.get("models", [])) or "unknown"
        label = led.get("label") or model
        lines.append("")
        lines.append(f"label    : {label}")
        lines.append(f"model    : {model}")
        lines.append(f"n        : {led.get('n')}")
        lines.append(f"asof_utc : {led.get('asof_utc')}")
        lines.append(f"source   : {led.get('source_file', '?')}")
        lines.append(f"root     : {led.get('root')}")
    return "\n".join(lines) + "\n"


def run(forecasts_glob="out/forecasts_*.json", asof_utc=None):
    """Commit every forecast file matching ``forecasts_glob``.

    Writes ``out/ledger_<model>.json`` per file plus ``out/ROOTS.txt``, and prints
    the roots -- that printed string is what a forecaster publishes. Returns the
    list of commitment dicts (empty if no forecast files exist, which is not an
    error).
    """
    if asof_utc is None:
        asof_utc = datetime.now(timezone.utc).isoformat()
    gpat = str(forecasts_glob)
    base, pat = (ROOT_DIR, gpat) if not pathlib.Path(gpat).is_absolute() else (
        pathlib.Path(pathlib.Path(gpat).anchor),
        str(pathlib.Path(gpat).relative_to(pathlib.Path(gpat).anchor)),
    )
    files = sorted(base.glob(pat))
    out_dir = ROOT_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"ledger: no forecast files matched {gpat!r} - nothing to commit.")
        print("        (run elicit.py first; a ledger over zero forecasts would claim nothing)")
        existing = sorted(out_dir.glob("ledger_*.json"))
        if existing:
            print(f"        {len(existing)} existing ledger(s) left untouched.")
        return []

    commits = []
    for f in files:
        try:
            records = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {f.name}: unreadable ({e})")
            continue
        if isinstance(records, dict):
            records = records.get("forecasts", records.get("records", []))
        if not isinstance(records, list):
            print(f"  ERR {f.name}: expected a JSON list of forecast records")
            continue
        label = _model_label(f, records)
        keep, excluded = committable(records)
        if not keep:
            print(f"  ERR {f.name}: no record carries a usable p - a ledger over zero claims"
                  " would commit to nothing; not writing one.")
            continue
        meta = {
            "source_file": f.name,
            "label": label,
            "committed_by": "ledger.py",
            "n_source_records": len(records),
            "excluded_indices": [e["index"] for e in excluded],
        }
        if excluded:
            meta["excluded"] = excluded[:50]
        try:
            led = commit(keep, asof_utc, meta=meta)
        except LedgerError as e:
            print(f"  ERR {f.name}: {e}")
            continue
        safe = _SAFE_LABEL.sub("-", label).strip("-") or "unknown"
        path = out_dir / f"ledger_{safe}.json"
        path.write_text(json.dumps(led, indent=2, sort_keys=True) + "\n")
        led["_path"] = str(path)
        commits.append(led)
        print(f"  committed {f.name}: n={led['n']:>4}  model={label}")
        if excluded:
            print(f"    excluded {len(excluded)} row(s) with no usable p "
                  f"(indices {[e['index'] for e in excluded][:10]}) - recorded in the ledger")
        print(f"    root {led['root']}")
        print(f"    -> {path.relative_to(ROOT_DIR)}")

    ledgers = []
    for p in sorted(out_dir.glob("ledger_*.json")):
        try:
            ledgers.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            continue
    roots_path = out_dir / "ROOTS.txt"
    roots_path.write_text(_roots_text(ledgers))
    print(f"  wrote {roots_path.relative_to(ROOT_DIR)} ({len(ledgers)} root(s)) - publish this file now,")
    print("  before anything resolves. An unpublished root is not a commitment.")
    return commits


def verify_cli(ledger_path):
    """Verify one ledger against the forecasts file it names. Returns True/False."""
    path = _resolve(ledger_path)
    try:
        led = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  cannot read {path}: {e}")
        return False
    src = led.get("source_file")
    if not src:
        model = led.get("model") or (led.get("models") or ["unknown"])[0]
        src = f"forecasts_{model}.json"
    fpath = ROOT_DIR / "out" / src
    try:
        records = json.loads(fpath.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  cannot read forecasts {fpath}: {e}")
        return False
    res = verify(path, records)
    print(f"  {path.name} vs {fpath.name}: {'MATCH' if res['ok'] else 'MISMATCH'}")
    print(f"    ledger root      {res.get('root_ledger')}")
    print(f"    recomputed root  {res.get('root_recomputed')}")
    if res["mismatched_leaves"]:
        print(f"    differing leaves {res['mismatched_indices']}")
        for m in res["mismatched_leaves"][:5]:
            print(f"      [{m['index']}] {m['tid_ledger']} -> {m['tid_record']}")
    for n in res["notes"]:
        print(f"    note: {n}")
    return res["ok"]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _synthetic_records(n=20, model="toy-model-1"):
    """Hand-checkable toy forecasts: p = (i+1)/(n+2), so every leaf is distinct."""
    return [
        {
            "tid": f"toy_family.T{i:02d}",
            "model": model,
            "p": round((i + 1) / (n + 2), 6),
            "rationale": f"toy rationale {i}",
            "asked_utc": "2026-07-26T00:00:00+00:00",
        }
        for i in range(n)
    ]


def self_test():
    """Assertions over canonicalization, the tree, proofs, and tamper detection."""
    import tempfile

    fails = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name} {detail}")
            fails.append(name)

    # ---- canonical(): what is and is not part of the claim ----------------
    r = {"tid": "f.A", "model": "m", "p": 0.34, "rationale": "because", "asked_utc": "2026-07-26"}
    check("canonical is exactly the claim fields",
          canonical(r) == b'{"model":"m","p":0.34,"tid":"f.A"}', canonical(r))
    r_cos = dict(r, rationale="COMPLETELY REWRITTEN", asked_utc="1999-01-01", extra={"z": 1})
    check("canonical ignores a changed rationale", leaf_hash(r) == leaf_hash(r_cos))
    check("canonical ignores key order",
          leaf_hash(r) == leaf_hash({"p": 0.34, "rationale": "x", "model": "m", "tid": "f.A"}))
    check("canonical does NOT ignore a changed p", leaf_hash(r) != leaf_hash(dict(r, p=0.35)))
    check("canonical does NOT ignore a changed p (0.01 nudge)",
          leaf_hash(r) != leaf_hash(dict(r, p=0.34 + 0.01)))
    check("canonical does NOT ignore a changed tid", leaf_hash(r) != leaf_hash(dict(r, tid="f.B")))
    check("canonical does NOT ignore a changed model", leaf_hash(r) != leaf_hash(dict(r, model="m2")))
    check("p type is canonicalized (0 == 0.0 == '0.0')",
          leaf_hash(dict(r, p=0)) == leaf_hash(dict(r, p=0.0)) == leaf_hash(dict(r, p="0.0")))
    for bad, why in [
        ({"tid": "a", "model": "m"}, "missing p"),
        ({"tid": "a", "model": "m", "p": 1.5}, "p>1"),
        ({"tid": "a", "model": "m", "p": float("nan")}, "p NaN"),
        ({"tid": "", "model": "m", "p": 0.5}, "empty tid"),
    ]:
        try:
            canonical(bad)
            check(f"canonical rejects {why}", False)
        except LedgerError:
            check(f"canonical rejects {why}", True)

    # ---- tree shape: hand-computed odd-node case -------------------------
    l0, l1, l2 = (hashlib.sha256(bytes([i])).hexdigest() for i in (0, 1, 2))
    check("odd-node convention == duplicate last (3 leaves)",
          merkle_root([l0, l1, l2]) == _node(_node(l0, l1), _node(l2, l2)))
    check("single leaf is its own root", merkle_root([l0]) == l0)
    check("two leaves", merkle_root([l0, l1]) == _node(l0, l1))
    check("empty tree is the sentinel", merkle_root([]) == EMPTY_ROOT)
    check("node hashes raw digests, not hex text",
          _node(l0, l1) == hashlib.sha256(bytes.fromhex(l0) + bytes.fromhex(l1)).hexdigest())
    check("root is order-sensitive", merkle_root([l0, l1]) != merkle_root([l1, l0]))

    # ---- proofs verify at every size and index ---------------------------
    proof_ok = True
    for n in (1, 2, 3, 5, 7, 8, 20, 21):
        recs = _synthetic_records(n)
        leaves = leaf_hashes(recs)
        root = merkle_root(leaves)
        for i in range(n):
            pr = merkle_proof(leaves, i)
            if not verify_proof(leaves[i], pr, root):
                proof_ok = False
                print(f"       proof failed at n={n} i={i}")
            if len(pr) != max(0, math.ceil(math.log2(n)) if n > 1 else 0):
                # depth check: ceil(log2(n)) levels for the duplicate-last tree
                proof_ok = False
                print(f"       unexpected proof depth n={n} i={i} len={len(pr)}")
    check("every merkle_proof verifies against the root (n=1,2,3,5,7,8,20,21)", proof_ok)

    recs20 = _synthetic_records(20)
    leaves20 = leaf_hashes(recs20)
    root20 = merkle_root(leaves20)
    pr7 = merkle_proof(leaves20, 7)
    check("proof rejects a wrong leaf", not verify_proof(leaves20[8], pr7, root20))
    check("proof rejects a wrong root", not verify_proof(leaves20[7], pr7, leaves20[0]))
    bad = [dict(s) for s in pr7]
    bad[0]["hash"] = l0
    check("proof rejects a tampered sibling", not verify_proof(leaves20[7], bad, root20))
    flipped_side = [dict(s) for s in pr7]
    flipped_side[0]["side"] = "left" if flipped_side[0]["side"] == "right" else "right"
    check("proof rejects a flipped side", not verify_proof(leaves20[7], flipped_side, root20))
    check("proof rejects malformed input", not verify_proof("nothex", pr7, root20))

    # ---- commit / verify round trip, then tamper -------------------------
    with tempfile.TemporaryDirectory() as td:
        lpath = pathlib.Path(td) / "ledger_toy-model-1.json"
        led = commit(recs20, "2026-07-26T00:00:00+00:00", meta={"source_file": "forecasts_toy.json"})
        lpath.write_text(json.dumps(led, indent=2, sort_keys=True))
        check("commit reports n and one model", led["n"] == 20 and led["model"] == "toy-model-1")
        check("commit root == merkle_root(leaf_hashes)", led["root"] == root20)
        check("commit records every leaf", led["leaf_hashes"] == leaves20)
        check("commit is self-describing",
              led["hash_alg"] == HASH_ALG
              and led["canonical_fields"] == list(CLAIM_FIELDS)
              and led["odd_node_convention"] == ODD_NODE_CONVENTION)
        check("commit keeps meta but core keys win", led["source_file"] == "forecasts_toy.json")

        v = verify(lpath, recs20)
        check("verify: untouched records match", v["ok"] and v["match"] and not v["mismatched_leaves"])

        cosmetic = [dict(x) for x in recs20]
        cosmetic[3]["rationale"] = "rewritten after the fact"
        check("verify: cosmetic edit still matches", verify(lpath, cosmetic)["ok"])

        TAMPER_I = 13
        tampered = [dict(x) for x in recs20]
        tampered[TAMPER_I] = dict(tampered[TAMPER_I], p=round(tampered[TAMPER_I]["p"] + 0.01, 6))
        vt = verify(lpath, tampered)
        check("verify: p+0.01 is detected", (not vt["ok"]) and (not vt["match"]))
        check(f"verify: names leaf index {TAMPER_I} and only that one",
              vt["mismatched_indices"] == [TAMPER_I], vt["mismatched_indices"])
        check("verify: reports the tampered tid",
              vt["mismatched_leaves"][0]["tid_record"] == recs20[TAMPER_I]["tid"]
              if vt["mismatched_leaves"] else False)

        reordered = list(recs20)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        vr = verify(lpath, reordered)
        check("verify: reordering is detected", not vr["ok"] and vr["mismatched_indices"] == [0, 1])

        # a failed elicitation (p=None) must not sink the whole commitment
        with_null = [dict(x) for x in recs20]
        with_null.insert(4, {"tid": "toy_family.BAD", "model": "toy-model-1", "p": None,
                             "rationale": "", "error": "parse failure"})
        keep, excl = committable(with_null)
        check("committable drops p=None and names the index",
              len(keep) == 20 and [e["index"] for e in excl] == [4], (len(keep), excl))
        check("committable keeps everything when nothing is null",
              committable(recs20) == (recs20, []))
        led_n = commit(keep, "2026-07-26T00:00:00+00:00",
                       meta={"n_source_records": 21, "excluded_indices": [4]})
        npath = pathlib.Path(td) / "ledger_null.json"
        npath.write_text(json.dumps(led_n, indent=2, sort_keys=True))
        check("commitment over the usable subset still equals the 20-leaf root",
              led_n["root"] == root20)
        check("verify re-derives the same exclusion and matches",
              verify(npath, with_null)["ok"])
        blanked = [dict(x) for x in with_null]
        blanked[9] = dict(blanked[9], p=None, error="blanked after the fact")
        vb = verify(npath, blanked)
        check("verify catches a real p blanked to null after commitment",
              not vb["ok"] and any("exclusion set differs" in n for n in vb["notes"]),
              vb["notes"])

        vd = verify(lpath, recs20[:-1])
        check("verify: a dropped record is detected (count first)",
              not vd["ok"] and vd["n_ledger"] == 20 and vd["n_records"] == 19)
        va = verify(lpath, recs20 + [_synthetic_records(1, "toy-model-1")[0]])
        check("verify: an appended record is detected", not va["ok"])
        check("verify: missing ledger file fails cleanly",
              not verify(pathlib.Path(td) / "nope.json", recs20)["ok"])

    print()
    if fails:
        print(f"FAIL  ledger.py self-test: {len(fails)} assertion(s) failed: {fails}")
        return False
    print("PASS  ledger.py self-test: canonicalization, odd-node tree, proofs, tamper localization")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Commit forecast records to a Merkle root.")
    ap.add_argument("--glob", default="out/forecasts_*.json", help="forecast files to commit")
    ap.add_argument("--asof", default=None, help="ISO-8601 asof_utc (default: now, UTC)")
    ap.add_argument("--self-test-only", action="store_true", help="run the self-test and stop")
    ap.add_argument("--no-self-test", action="store_true", help="skip the self-test, just commit")
    ap.add_argument("--verify", metavar="LEDGER", default=None,
                    help="verify a ledger against the forecasts file it names")
    a = ap.parse_args(argv)

    if a.verify:
        return 0 if verify_cli(a.verify) else 1

    ok = True
    if not a.no_self_test:
        print("ledger.py self-test")
        ok = self_test()
        if a.self_test_only or not ok:
            return 0 if ok else 1
        print()
    print(f"ledger.py commit pass (glob={a.glob!r})")
    run(a.glob, a.asof)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
