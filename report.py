"""Build reports/report_<ts>.html from whatever out/*.json exists.

Tolerant by design: every section renders "not run" rather than crashing, so a
partial pipeline still produces a readable report. Charts are hand-built inline
SVG — no JS libraries, no network, works from file://.
"""
import html
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]


# ---------------------------------------------------------------- data loading

def load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def dig(obj, *paths, default=None):
    """First non-None value among dotted paths. Survives agent-to-agent key drift."""
    for path in paths:
        cur = obj
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
                cur = cur[int(k)]
            else:
                cur = None
                break
        if cur is not None:
            return cur
    return default


def esc(s):
    return html.escape(str(s))


def fmt(x, nd=3):
    if x is None:
        return "&mdash;"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


# ---------------------------------------------------------------- chart primitives

def hbar(rows, width=760, rowh=26, pad_left=290, fmtv=lambda v: f"{v:.3f}",
         color=SERIES[0], vmax=None, title=""):
    """Sorted horizontal bars with direct value labels. rows: [(label, value, sublabel)]."""
    if not rows:
        return "<p class='none'>no data</p>"
    vmax = vmax or max(abs(v) for _, v, *_ in rows) or 1.0
    h = rowh * len(rows) + 28
    plotw = width - pad_left - 90
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' aria-label='{esc(title)}' class='chart'>"]
    for i, row in enumerate(rows):
        label, v = row[0], row[1]
        sub = row[2] if len(row) > 2 else ""
        y = i * rowh + 8
        bw = max(2.0, abs(v) / vmax * plotw)
        out.append(
            f"<text x='{pad_left - 10}' y='{y + rowh * 0.62}' class='tick' text-anchor='end'>"
            f"{esc(label)}</text>")
        out.append(
            f"<rect x='{pad_left}' y='{y + 4}' width='{bw:.1f}' height='{rowh - 12}' rx='4' "
            f"fill='{color}'><title>{esc(label)}: {fmtv(v)} {esc(sub)}</title></rect>")
        out.append(
            f"<text x='{pad_left + bw + 8}' y='{y + rowh * 0.62}' class='val'>{fmtv(v)}"
            f"{(' ' + esc(sub)) if sub else ''}</text>")
    out.append(f"<line x1='{pad_left}' y1='4' x2='{pad_left}' y2='{h - 20}' class='axis'/>")
    out.append("</svg>")
    return "".join(out)


def grouped_bars(cats, series, width=760, height=240, fmtv=lambda v: f"{v:g}", title=""):
    """cats: [label]. series: [(name, [values aligned to cats])]."""
    if not cats or not series:
        return "<p class='none'>no data</p>"
    vmax = max([max(vals) for _, vals in series] + [1])
    padl, padb, padt = 46, 58, 12
    plotw, ploth = width - padl - 12, height - padb - padt
    gw = plotw / len(cats)
    bw = min(30.0, gw / (len(series) + 0.6))
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{esc(title)}' class='chart'>"]
    for g in range(5):
        gy = padt + ploth - ploth * g / 4
        out.append(f"<line x1='{padl}' y1='{gy:.1f}' x2='{width - 12}' y2='{gy:.1f}' class='grid'/>")
        out.append(f"<text x='{padl - 8}' y='{gy + 4:.1f}' class='tick' text-anchor='end'>"
                   f"{vmax * g / 4:g}</text>")
    for ci, cat in enumerate(cats):
        x0 = padl + ci * gw
        for si, (name, vals) in enumerate(series):
            v = vals[ci] if ci < len(vals) else 0
            bh = (v / vmax) * ploth if vmax else 0
            x = x0 + gw / 2 - (len(series) * bw + (len(series) - 1) * 2) / 2 + si * (bw + 2)
            y = padt + ploth - bh
            if bh > 0.5:
                out.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{bh:.1f}' "
                           f"rx='4' fill='{SERIES[si % len(SERIES)]}'>"
                           f"<title>{esc(name)} · {esc(cat)}: {fmtv(v)}</title></rect>")
                out.append(f"<text x='{x + bw / 2:.1f}' y='{y - 4:.1f}' class='val' "
                           f"text-anchor='middle'>{fmtv(v)}</text>")
        words = str(cat).replace("_", " ").split()
        mid = (len(words) + 1) // 2
        l1, l2 = " ".join(words[:mid]), " ".join(words[mid:])
        out.append(f"<text x='{x0 + gw / 2:.1f}' y='{padt + ploth + 16}' class='tick' "
                   f"text-anchor='middle'>{esc(l1)}</text>")
        if l2:
            out.append(f"<text x='{x0 + gw / 2:.1f}' y='{padt + ploth + 29}' class='tick' "
                       f"text-anchor='middle'>{esc(l2)}</text>")
    out.append(f"<line x1='{padl}' y1='{padt + ploth}' x2='{width - 12}' y2='{padt + ploth}' "
               f"class='axis'/>")
    out.append("</svg>")
    return "".join(out)


def dumbbell(rows, width=760, rowh=24, pad_left=290, title=""):
    """rows: [(label, quoted, repaired)] on a 0..1 probability axis."""
    if not rows:
        return "<p class='none'>no data</p>"
    h = rowh * len(rows) + 34
    plotw = width - pad_left - 60
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' aria-label='{esc(title)}' class='chart'>"]
    for g in range(6):
        gx = pad_left + plotw * g / 5
        out.append(f"<line x1='{gx:.1f}' y1='2' x2='{gx:.1f}' y2='{h - 26}' class='grid'/>")
        out.append(f"<text x='{gx:.1f}' y='{h - 12}' class='tick' text-anchor='middle'>"
                   f"{g / 5:.1f}</text>")
    for i, (label, a, b) in enumerate(rows):
        y = i * rowh + 12
        xa, xb = pad_left + a * plotw, pad_left + b * plotw
        out.append(f"<text x='{pad_left - 10}' y='{y + 4}' class='tick' text-anchor='end'>"
                   f"{esc(label)}</text>")
        out.append(f"<line x1='{xa:.1f}' y1='{y}' x2='{xb:.1f}' y2='{y}' class='connector'/>")
        out.append(f"<circle cx='{xa:.1f}' cy='{y}' r='5' fill='{SERIES[0]}' class='ring'>"
                   f"<title>{esc(label)} quoted {a:.3f}</title></circle>")
        out.append(f"<circle cx='{xb:.1f}' cy='{y}' r='5' fill='{SERIES[1]}' class='ring'>"
                   f"<title>{esc(label)} repaired {b:.3f}</title></circle>")
    out.append("</svg>")
    return "".join(out)


def legend(items):
    sw = "".join(
        f"<span class='lg'><i style='background:{SERIES[i % len(SERIES)]}'></i>{esc(n)}</span>"
        for i, n in enumerate(items))
    return f"<div class='legend'>{sw}</div>"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c if isinstance(c, str) and c.startswith('<') else esc(c)}</td>"
                                  for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def details(summary, body):
    return f"<details><summary>{esc(summary)}</summary>{body}</details>"


CSS = """
:root{color-scheme:light}
.viz-root{
 --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
 --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
 --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
 --critical:#d03b3b; --good:#0ca30c; --warning:#fab219;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
 color-scheme:dark;
 --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
 --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
 --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}}
:root[data-theme="dark"] .viz-root{
 color-scheme:dark;
 --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
 --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
 --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:44px 0 6px;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 4px;color:var(--text-secondary);font-weight:600}
p{margin:8px 0}
.sub{color:var(--text-secondary);font-size:14px;margin:0 0 4px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
 padding:18px 20px;margin:14px 0}
.hero{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0 6px}
.tile{flex:1 1 190px;background:var(--surface-1);border:1px solid var(--border);
 border-radius:12px;padding:16px 18px}
.tile .k{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-size:30px;font-weight:650;margin:4px 0 0;letter-spacing:-.02em}
.tile .n{font-size:13px;color:var(--text-secondary)}
.chart{width:100%;height:auto;display:block;margin:6px 0 2px;overflow:visible}
text{font:11px system-ui,-apple-system,sans-serif}
.tick{fill:var(--muted)}
.val{fill:var(--text-secondary);font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.connector{stroke:var(--axis);stroke-width:2}
.ring{stroke:var(--surface-1);stroke-width:2}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 2px;font-size:13px;
 color:var(--text-secondary)}
.lg{display:flex;align-items:center;gap:6px}
.lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--grid);
 font-variant-numeric:tabular-nums;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
 letter-spacing:.05em}
td.l,th.l{white-space:normal}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.root{word-break:break-all;color:var(--text-secondary)}
details{margin:10px 0}
summary{cursor:pointer;color:var(--text-secondary);font-size:14px}
.none{color:var(--muted);font-style:italic;font-size:14px}
.pill{display:inline-block;font-size:11px;padding:2px 9px;border-radius:99px;
 border:1px solid var(--border);color:var(--text-secondary);margin-right:6px}
.bad{color:var(--critical);font-weight:600}
.ok{color:var(--good);font-weight:600}
.warn{color:var(--warning);font-weight:600}
.scroll{overflow-x:auto}
.foot{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--grid);
 padding-top:14px}
"""


# ---------------------------------------------------------------- sections

def short(tid):
    return tid.split(".", 1)[-1][:26]


def sec_hero(coh, db, leak, base):
    tiles = []
    if db:
        worst = max(db["models"].items(), key=lambda kv: kv[1]["mean_dollars_per_100"])
        tiles.append(("Guaranteed loss, worse model", f"${worst[1]['mean_dollars_per_100']:.2f}",
                      f"per $100 staked, mean over {worst[1]['n_families_scored']} families "
                      f"&mdash; {esc(worst[0])}"))
        n_arb = sum(m["n_arbitrage"] for m in db["models"].values())
        n_tot = sum(m["n_families_scored"] for m in db["models"].values())
        tiles.append(("Families arbitrageable", f"{n_arb}/{n_tot}",
                      "an arbitrage exists at the forecaster's own quoted prices"))
    if leak and dig(leak, "date_probe.n_already_resolved") is not None:
        dp = leak["date_probe"]
        tiles.append(("Hindsight-free", f"{dp['n_already_resolved']}/{dp['n_families']}",
                      "families the model reads as already resolved "
                      f"&mdash; {dp['frac_within_2_months_of_truth'] * 100:.0f}% dated to within 2 months of now"))
    if leak and dig(leak, "contamination_gap.implied_backtest_discount") is not None:
        cg = leak["contamination_gap"]
        tiles.append(("Backtest discount", f"{cg['implied_backtest_discount'] * 100:.0f}%",
                      "of the model's sharpness on resolved events does not transfer to open ones"))
    if base and base.get("arms"):
        r = max(base["arms"].values(), key=lambda v: v.get("ratio_to_floor") or 0)
        if r.get("ratio_to_floor"):
            tiles.append(("Signal vs sampling noise", f"{r['ratio_to_floor']:.1f}&times;",
                          "the arbitrage a perfectly coherent forecaster with the same "
                          "test&ndash;retest jitter would show"))
    return "<div class='hero'>" + "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div>"
        f"<div class='n'>{n}</div></div>" for k, v, n in tiles) + "</div>"


def sec_coherence(coh, db):
    if not coh or not coh.get("results"):
        return "<p class='none'>coherence audit not run</p>"
    models = sorted(coh["aggregates"])
    rows = []
    for m in models:
        a = coh["aggregates"][m]
        dbm = (db or {}).get("models", {}).get(m, {})
        rows.append([m, a["n_families"], f"{a['mean_L2']:.4f}", f"{a['median_L2']:.4f}",
                     f"{a['max_L2']:.4f}", f"{a['mean_abs_repair']:.4f}",
                     f"${dbm.get('mean_dollars_per_100', 0):.2f}",
                     f"${dbm.get('max_dollars_per_100', 0):.2f}", a["n_violations"]])
    tbl = table(["model", "families", "mean L2", "median L2", "max L2", "mean repair",
                 "mean $/100", "max $/100", "violations"], rows)

    axioms = sorted({k for m in models for k in coh["aggregates"][m]["violations_by_axiom"]})
    series = [(m, [coh["aggregates"][m]["violations_by_axiom"].get(a, 0) for a in axioms])
              for m in models]
    chart = grouped_bars([a.replace("_", " ") for a in axioms], series,
                         title="axiom violations by type")

    per = {}
    for r in db["results"] if db else []:
        per.setdefault(r["model"], []).append(r)
    fam_charts = []
    for m in models:
        rs = sorted(per.get(m, []), key=lambda r: -r["dollars_per_100"])[:12]
        if rs:
            fam_charts.append(
                f"<h3>{esc(m)} &mdash; worst 12 families</h3>"
                + hbar([(r["family_id"][:40], r["dollars_per_100"], "") for r in rs],
                       fmtv=lambda v: f"${v:.2f}", color=SERIES[models.index(m) % 3],
                       title=f"{m} arbitrage per family"))
    return tbl + legend(models) + chart + "".join(fam_charts)


def sec_certificate(db):
    if not db or not db.get("results"):
        return ""
    worst = max(db["results"], key=lambda r: r["dollars_per_100"])
    lines = "\n".join(worst["certificate"])
    return (f"<p>The single worst family in the run &mdash; <code>{esc(worst['family_id'])}</code>, "
            f"{esc(worst['model'])}. This book is constructed from the forecaster's own quoted "
            f"prices and pays out on <strong>every one of the {worst['n_atoms']} logically "
            f"possible outcomes</strong>:</p>"
            f"<div class='card'><pre class='mono' style='white-space:pre-wrap;margin:0'>"
            f"{esc(lines)}</pre></div>")


def sec_repair(coh, db):
    if not coh or not coh.get("results") or not db:
        return ""
    worst = max(db["results"], key=lambda r: r["dollars_per_100"])
    row = next((r for r in coh["results"]
                if r["family_id"] == worst["family_id"] and r["model"] == worst["model"]), None)
    if not row:
        return ""
    rows = [(short(t), p, f) for t, p, f in zip(row["tids"], row["prices"], row["fitted"])]
    return (f"<p>What the repair does to that family. Blue is what the forecaster said; orange is "
            f"the nearest set of numbers that is internally consistent. Mean move "
            f"{row['mean_abs_repair']:.3f}, largest {row['max_abs_repair']:.3f}.</p>"
            + legend(["quoted", "repaired (coherent)"])
            + dumbbell(rows, title="quoted vs repaired probability"))


def sec_baselines(base):
    if not base or not base.get("arms"):
        return ("<p class='none'>Noise floor not yet computed &mdash; run "
                "<code>elicit.py --repeat 3</code> then <code>baselines.py</code>. "
                "Until then the arbitrage magnitudes above are unvalidated against sampling "
                "variance and should not be quoted.</p>")
    rows, cats, s_obs, s_floor, s_ceil = [], [], [], [], []
    for key, r in sorted(base["arms"].items()):
        verdict = ("<span class='ok'>above noise</span>" if (r.get("ratio_to_floor") or 0) >= 1.5
                   else "<span class='bad'>NOT above noise</span>")
        rows.append([key, r["families_covered"], f"{r['retest_sd_used']:.3f}",
                     f"${r['observed_dollars_per_100']:.2f}",
                     f"${r['floor_dollars_per_100']:.2f}",
                     f"${r['ceiling_dollars_per_100']:.2f}",
                     f"{r['ratio_to_floor']:.2f}x" if r.get("ratio_to_floor") else "&mdash;",
                     f"{r['p_noise_explains']:.3f}", verdict])
        cats.append(key.replace("|", " "))
        s_obs.append(r["observed_dollars_per_100"])
        s_floor.append(r["floor_dollars_per_100"])
        s_ceil.append(r["ceiling_dollars_per_100"])
    return (table(["arm", "families", "retest SD", "observed", "noise floor", "ceiling",
                   "vs floor", "p(noise)", "verdict"], rows)
            + legend(["observed", "noise floor (coherent + own jitter)", "ceiling (text-blind random)"])
            + grouped_bars(cats, [("observed", s_obs), ("noise floor", s_floor),
                                  ("ceiling", s_ceil)],
                           fmtv=lambda v: f"${v:.1f}", title="observed vs floor vs ceiling"))


def sec_leakage(leak):
    if not leak:
        return "<p class='none'>leakage audit not run</p>"
    dp, cg = leak.get("date_probe", {}), leak.get("contamination_gap", {})
    out = []
    if dp.get("n_families"):
        cert = ("<span class='ok'>CERTIFIED HINDSIGHT-FREE</span>"
                if dp.get("certified_hindsight_free") else
                "<span class='bad'>NOT CERTIFIED</span>")
        out.append(f"<p>{cert} &mdash; {esc(dp['certificate_rule'])}</p>")
        out.append(table(["probe", "value"], [
            ["families probed", dp["n_families"]],
            ["reads as already resolved", f"{dp['n_already_resolved']} "
             f"({dp['frac_already_resolved'] * 100:.0f}%)"],
            ["median dating error vs 2026-07", f"{dp['months_off_median']:.1f} months"],
            ["dated within 2 months of now", f"{dp['frac_within_2_months_of_truth'] * 100:.0f}%"],
            ["written-date histogram",
             ", ".join(f"{k}: {v}" for k, v in dp.get("written_estimate_hist", {}).items())]]))
    if cg.get("past"):
        p, f = cg["past"], cg["future"]
        out.append("<h3>Recall versus forecasting</h3>")
        out.append(table(["", "resolved controls (2024&ndash;25)", "open questions (this set)"], [
            ["n", p.get("n", cg.get("n_controls_scored")), f.get("n")],
            ["Brier", fmt(p.get("brier"), 4), "&mdash; unresolved"],
            ["skill vs uninformative", fmt(p.get("brier_skill_vs_uninformative"), 3), "&mdash;"],
            ["accuracy", fmt(p.get("accuracy"), 3), "&mdash;"],
            ["mean confidence |p&minus;0.5|", fmt(p.get("mean_confidence"), 4),
             fmt(f.get("mean_confidence"), 4)]]))
        out.append(f"<p><strong>Confidence gap "
                   f"{cg['gap_confidence_past_minus_future']:+.3f}</strong> &rarr; implied backtest "
                   f"discount {cg['implied_backtest_discount'] * 100:.0f}%. "
                   f"{esc(cg.get('interpretation', ''))}</p>")
        out.append(f"<p class='sub'>{esc(cg.get('clean_test_note', ''))}</p>")
    return "".join(out)


def sec_ledger():
    roots = (OUT / "ROOTS.txt")
    led = sorted(OUT.glob("ledger_*.json"))
    if not led:
        return "<p class='none'>nothing committed yet</p>"
    rows = []
    for p in led:
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rows.append([p.name, d.get("n"), f"<span class='root mono'>{esc(d.get('root', ''))}</span>",
                     d.get("asof_utc", "")])
    body = table(["ledger", "leaves", "merkle root (sha256)", "sealed at (UTC)"], rows)
    sc = load("scoring.json")
    note = ""
    if roots.exists():
        note += ("<p class='sub'>out/ROOTS.txt is the publishable artefact: it carries the roots "
                 "plus the leaf/node/odd-node rules, so a stranger can recompute them from the "
                 "forecast files without reading this repo's source.</p>")
    note += ("<p>Scoring is <strong>armed but deliberately not runnable</strong>: every question "
             "in this set resolves in the future, so there are no outcomes to score. "
             "The proper-scoring machinery (Brier, log score, Murphy decomposition, reliability "
             "curve with Wilson intervals, skill-vs-base-rate, horizon decay) is implemented and "
             "self-tested against a synthetic calibrated forecaster; it turns on when outcomes "
             "are written to <code>out/resolutions.json</code>.</p>")
    if isinstance(sc, dict) and sc.get("earliest_resolve_by"):
        note += f"<p class='sub'>earliest resolution: {esc(sc['earliest_resolve_by'])}</p>"
    return body + note


def sec_questions(fams):
    rows = []
    for f in fams:
        rows.append([f["family_id"], f["domain"], len(f["base_events"]), len(f["targets"]),
                     min(e["resolve_by"] for e in f["base_events"].values()),
                     ", ".join(c["type"] for c in f.get("constraints", [])) or "&mdash;"])
    return table(["family", "domain", "events", "targets", "earliest resolve_by", "constraints"],
                 rows)


def main():
    coh, db, leak, base = (load("coherence.json"), load("dutchbook.json"),
                           load("leakage.json"), load("baselines.json"))
    import algebra
    fams, errs = algebra.load_all(str(ROOT / "questions"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    n_t = sum(len(f["targets"]) for f in fams)

    body = f"""
<div class='viz-root'><div class='wrap'>
<h1>Forecast integrity audit</h1>
<p class='sub'>{len(fams)} question families &middot; {n_t} targets &middot;
{esc(time.strftime('%Y-%m-%d %H:%M'))} &middot; every event resolves after this report was written</p>

{sec_hero(coh, db, leak, base)}

<div class='card'>
<p style='margin-top:0'><strong>What this measures.</strong> A forecasting system cannot be
evaluated on accuracy until its questions resolve. This audit scores one anyway, using three things
that are checkable on day zero: whether its probabilities obey the probability axioms, whether an
arbitrage can be constructed against them, and whether its apparent skill is recall of events it
already knows about.</p>
<p style='margin-bottom:0'>The forecaster is asked each question in a <em>separate process with no
memory of the others</em>, so it cannot paper over an inconsistency by looking at what it just said.
Its answers are then sealed in a Merkle commitment before any analysis touches them.</p>
</div>

<h2>1. Coherence &mdash; does the forecaster contradict itself?</h2>
<p class='sub'>Two independent auditors: a projection onto the coherent set (L2 = distance from the
nearest internally consistent forecast) and a linear program that tries to build a guaranteed-profit
book. They share no code beyond the event algebra, and they agree on every family.</p>
{sec_coherence(coh, db)}

<h2>2. The worst book, in full</h2>
{sec_certificate(db)}
{sec_repair(coh, db)}

<h2>3. Is it above sampling noise?</h2>
<p class='sub'>Random uniform quotes are arbitrageable ~100% of the time, so the fraction of
arbitrageable families proves nothing &mdash; the magnitude has to be bracketed. The floor is the
model's own repaired forecast re-perturbed at its measured test&ndash;retest SD: the arbitrage a
perfectly coherent forecaster would show purely from failing to repeat itself.</p>
{sec_baselines(base)}

<h2>4. Contamination &mdash; is this forecasting or recall?</h2>
{sec_leakage(leak)}

<h2>5. The sealed commitment</h2>
{sec_ledger()}

<h2>6. The question set</h2>
<p class='sub'>{len(fams)} families, {n_t} targets, {len(errs)} validation errors. Every base event
was web-verified as unresolved at authoring time and resolves between 2026-09-01 and 2027-06-30.</p>
<div class='scroll'>{sec_questions(fams)}</div>

<h2>Limitations</h2>
<ul>
<li>Coherence is necessary, not sufficient. A forecaster that answers 0.5 everywhere is perfectly
coherent and perfectly useless. This audit is a filter, not a ranking.</li>
<li>The contamination gap is an <strong>upper bound</strong> on leakage, not a point estimate: the
resolved controls are famous events while the open questions are deliberately near-even, so some of
the gap is the past genuinely being more knowable. The clean version is the matched-difficulty
pre/post-cutoff split, which becomes runnable when this set resolves.</li>
<li>The date probe partly measures where the model's knowledge cutoff sits, not only leakage.</li>
<li>Accuracy, calibration and horizon decay are unmeasured by construction. That is the point of the
commitment: they become measurable later, on forecasts that were sealed first.</li>
</ul>

<p class='foot'>Generated by report.py &middot; charts are inline SVG, no external requests &middot;
all numbers reproducible via <code>python3 run_all.py</code></p>
</div></div>"""

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Forecast integrity audit</title><style>{CSS}</style></head>"
           f"<body>{body}</body></html>")
    p = REPORTS / f"report_{ts}.html"
    p.write_text(doc)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    main()
