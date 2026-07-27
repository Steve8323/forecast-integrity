"""Cached headless-Claude caller. Uses `claude -p` (existing CLI auth, no API key).

Every unique (model, variant, prompt) triple is called once and cached by sha256 on
disk under ./cache, so nothing is ever billed twice (standing rule for paid /
rate-limited APIs). Delete a cache file to force a re-ask; never delete the dir.

`variant` folds in the things that change the answer but not the prompt text: the
tool policy, the effort level, and a hash of the system prompt. Without it the
retrieval-augmented arm would silently read the pure-prior arm's cached answers.

Isolation, which matters because the callee is the subject of an audit:
  * cwd is this repo dir, and there is no CLAUDE.md here or in any parent, so no
    memory/instruction file reaches the forecaster.
  * `--safe-mode` drops CLAUDE.md discovery, skills, plugins, hooks, MCP servers,
    custom agents and output styles (auth and model selection still work normally).
    It also stops the user's global Stop hook from firing a desktop notification
    once per call.
  * `--system-prompt` replaces the Claude Code agent system prompt outright, so the
    forecaster never sees the harness, the cwd listing, or the repo's git status.
  * `--no-session-persistence` keeps 200-call runs from littering ~/.claude.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
WORKDIR = str(HERE)  # no CLAUDE.md here or above it: nothing leaks into the callee

# Pure-prior arm: no retrieval, no filesystem, no shell, no subagents.
DENY_ALL = (
    "WebSearch,WebFetch,Bash,BashOutput,KillShell,Read,Write,Edit,MultiEdit,"
    "NotebookEdit,Glob,Grep,Task,TodoWrite,ExitPlanMode,Skill,SlashCommand"
)
# Retrieval arm: web only, still no filesystem/shell/subagents.
ALLOW_SEARCH = "WebSearch,WebFetch"
DENY_EXCEPT_SEARCH = (
    "Bash,BashOutput,KillShell,Read,Write,Edit,MultiEdit,NotebookEdit,"
    "Glob,Grep,Task,TodoWrite,ExitPlanMode,Skill,SlashCommand"
)

#  FROZEN. SYSTEM_PROMPT and ARGV_VERSION below are hashed into every cache key, and
#  other modules (leakage.py) already have paid replies sitting in ./cache under them.
#  Editing either string silently orphans every one of those and re-bills the lot.
SYSTEM_PROMPT = (
    "You are a careful, well-calibrated forecaster. You answer a single probability "
    "question at a time, in the exact output format the question asks for, with no "
    "preamble and no commentary."
)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "low"  # the user's global effortLevel is xhigh; that is far too slow here

# Bump when _argv changes in a way that could change the answer: it is folded into the
# cache key, so old replies are retired instead of silently reused under a new config.
ARGV_VERSION = "2"

# API failures that retrying cannot fix (bad model name, auth). Fail fast rather than
# spend 6 backoff rounds x N targets discovering the same thing.
_FATAL_STATUS = {400, 401, 403, 404, 405}

_LOCK = threading.Lock()
STATS = {"hits": 0, "calls": 0, "retries": 0, "cost_usd": 0.0}


def reset_stats():
    with _LOCK:
        STATS.update({"hits": 0, "calls": 0, "retries": 0, "cost_usd": 0.0})


class FatalCallError(RuntimeError):
    """A `claude -p` failure that retrying will not fix."""


def _variant(tools, effort, system_prompt):
    sp = hashlib.sha256((system_prompt or "").encode()).hexdigest()[:8]
    return (f"cfg={ARGV_VERSION};tools={'search' if tools else 'off'};"
            f"effort={effort};sp={sp}")


def cache_path(prompt, model=DEFAULT_MODEL, tools=False, effort=DEFAULT_EFFORT,
               system_prompt=SYSTEM_PROMPT):
    v = _variant(tools, effort, system_prompt)
    key = hashlib.sha256(f"{model}\x00{v}\x00{prompt}".encode()).hexdigest()
    return CACHE / f"{key}.json"


def is_cached(prompt, model=DEFAULT_MODEL, **kw):
    return cache_path(prompt, model, **kw).exists()


def _atomic_write(path, text):
    """Write via a unique temp file + rename: 8 workers (and other modules) share ./cache,
    and a half-written cache file would poison a later run for free."""
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _argv(model, tools, effort, system_prompt):
    argv = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--safe-mode",
        "--no-session-persistence",
        "--effort", effort,
        "--system-prompt", system_prompt,
    ]
    if tools:
        # --tools restricts the built-in set (so no other schema is even sent);
        # --allowedTools pre-approves these two so -p never stalls on permission.
        argv += ["--tools", ALLOW_SEARCH,
                 "--allowedTools", ALLOW_SEARCH,
                 "--disallowedTools", DENY_EXCEPT_SEARCH]
    else:
        # `--tools ""` sends no tool schemas at all: cheaper, and the forecaster has
        # no way to notice it is inside a harness. --disallowedTools belt-and-braces.
        argv += ["--tools", "", "--disallowedTools", DENY_ALL]
    return argv


def call(prompt, model=DEFAULT_MODEL, tag="", tools=False, effort=DEFAULT_EFFORT,
         system_prompt=SYSTEM_PROMPT, timeout=None, attempts=6, verbose=True):
    """One cached headless-Claude turn. Returns the model's reply text."""
    cpath = cache_path(prompt, model, tools=tools, effort=effort, system_prompt=system_prompt)
    if cpath.exists():
        with _LOCK:
            STATS["hits"] += 1
        return json.loads(cpath.read_text())["result"]

    if timeout is None:
        timeout = 900 if tools else 300
    argv = _argv(model, tools, effort, system_prompt)
    last = None
    for attempt in range(attempts):
        try:
            proc = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                                  timeout=timeout, cwd=WORKDIR)
            if not proc.stdout.strip():
                raise RuntimeError(
                    f"claude produced no stdout (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:300]}")
            obj = json.loads(proc.stdout)
            if obj.get("is_error") or obj.get("subtype") != "success":
                msg = f"claude error: {str(obj)[:300]}"
                if obj.get("api_error_status") in _FATAL_STATUS:
                    raise FatalCallError(msg)
                raise RuntimeError(msg)
            _atomic_write(cpath, json.dumps({
                "tag": tag, "model": model,
                "variant": _variant(tools, effort, system_prompt),
                "result": obj["result"],
                "cost_usd": obj.get("total_cost_usd"),
                "duration_ms": obj.get("duration_ms"),
                "num_turns": obj.get("num_turns"),
                "usage": obj.get("usage"),
            }))
            with _LOCK:
                STATS["calls"] += 1
                STATS["cost_usd"] += obj.get("total_cost_usd") or 0.0
            return obj["result"]
        except FatalCallError:
            raise  # bad model / auth: no amount of waiting helps
        except Exception as e:  # noqa: BLE001 - transient CLI/network failures
            last = e
            if attempt == attempts - 1:
                break
            with _LOCK:
                STATS["retries"] += 1
            # 529s and connection failures arrive in minutes-long bursts: wait them out.
            transient = any(s in str(e) for s in ("529", "overloaded", "Overloaded",
                                                  "onnect", "socket", "Socket", "rate"))
            wait = (30 if transient else 10) * (attempt + 1)
            if verbose:
                print(f"  retry {tag or cpath.stem[:8]} in {wait}s after: {str(e)[:200]}",
                      flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{tag or cpath.stem[:8]}: all {attempts} attempts failed: {last}")


def call_many(jobs, workers=8):
    """jobs: list of (prompt, model, tag) or dicts of kwargs. Results in order."""
    def run(j):
        return call(**j) if isinstance(j, dict) else call(*j)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, jobs))


def extract_json(text):
    """Parse the first JSON object/array in a model reply (handles ``` fences)."""
    s = (text or "").strip()
    if "```" in s:
        for p in s.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p[:1] in ("[", "{"):
                s = p
                break
    starts = [i for i in (s.find("["), s.find("{")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON found in reply: {s[:160]!r}")
    start = min(starts)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    return json.loads(s[start:])


def cache_summary():
    n = len(list(CACHE.glob("*.json")))
    return f"{n} cached replies in {CACHE}"


if __name__ == "__main__":
    print(cache_summary())
    print("stats:", STATS)
