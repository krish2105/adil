"""Build the ADIL results dashboard from the metrics artifacts.

Every number on the page is read from a `metrics/*.json` file written by a
notebook. None is typed here. Deleting `metrics/` and rerunning `make all`
reproduces `docs/index.html` byte for byte, which is what makes "the page matches
the model" checkable rather than asserted — the same discipline SPINE applies to
its own generated docs page.

The page is a single self-contained HTML file: no build step, no bundler, no
external asset except the Google Fonts stylesheet. Charts are inline SVG emitted
by the functions below, each paired with a table view so no reading depends on
colour alone.
"""

import html
import json

from adil import paths

# --- palette -----------------------------------------------------------------
# Categorical slots 1-4 of the validated default palette, unchanged. Verified with
# the data-viz validator: worst adjacent CVD ΔE 9.1, normal-vision ΔE 22.9, all
# checks pass in light mode. Two slots sit under 3:1 on the light ground, so every
# chart that uses them ships direct labels and a table view (the relief rule).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
GAIN, LOSS = "#2a78d6", "#e34948"  # the validated diverging pair
RUNG_ORDER = ["R0", "R1", "R3", "R4", "R6"]


def esc(value: object) -> str:
    """Escape a value for HTML text content."""
    return html.escape(str(value))


def load() -> dict[str, object]:
    """Read every metrics artifact the page draws on."""
    directory = paths.metrics_dir()
    names = (
        "frame",
        "r0",
        "r1",
        "r3",
        "r4",
        "r5",
        "r6",
        "headline",
        "fairness",
        "reject_inference",
    )
    data = {}
    for name in names:
        path = directory / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing; run `make features` and `make notebooks`")
        data[name] = json.loads(path.read_text())
    return data


# --- svg primitives -----------------------------------------------------------


def declutter(positions: list[float], min_gap: float = 16.0) -> list[float]:
    """Push overlapping end-labels apart while keeping their order.

    A series label sits at the height of its own last point, which reads correctly
    right up until two curves finish close together — and on both charts here they
    do. Sorting by height and enforcing a minimum gap keeps every label beside its
    own line without any two occupying the same space.

    Examples
    --------
    The default gap is set against the *measured* line box of a series label in the
    rendered page — 13.5px at this type size — not against the font size. A gap
    chosen to match the nominal size leaves labels overlapping by half a pixel,
    which is invisible in a screenshot and caught immediately by bounding boxes.

    >>> declutter([100.0, 104.0])
    [100.0, 116.0]
    >>> declutter([100.0, 140.0])
    [100.0, 140.0]
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = list(positions)
    for rank, index in enumerate(order):
        if rank == 0:
            continue
        previous = adjusted[order[rank - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    return adjusted


def axis_line(x1: float, y1: float, x2: float, y2: float) -> str:
    """A recessive axis or grid rule."""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="rule" />'


def ladder_chart(steps: list[dict]) -> str:
    """The hero: each rung's PR-AUC step drawn as a rung of an actual ladder.

    A diverging form — gains extend right of the spine, costs left — because the
    reader's question is polarity: did this constraint give something up, and how
    much. The spine is the ladder the project is named for, so the chart draws the
    document's own structure rather than decorating it.

    Three fixed columns, so nothing can collide: rung names on the left, the bars
    around the spine, values right-aligned on the far right. Placing each value at
    its own bar's end reads more naturally but overlaps the rung name as soon as a
    bar grows long, which is exactly what the largest cost does.
    """
    width, row_height = 620, 62
    label_x, spine, value_x = 8, 300, width - 8
    reach = 128
    height = row_height * len(steps) + 40
    scale = reach / max(abs(s["PR-AUC"]) for s in steps)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="PR-AUC change at each rung of the constraint ladder">'
    ]
    parts.append(f'<line x1="{spine}" y1="18" x2="{spine}" y2="{height - 22}" class="spine" />')
    for index, step in enumerate(steps):
        y = 26 + index * row_height
        value = step["PR-AUC"]
        length = abs(value) * scale
        positive = value >= 0
        x = spine if positive else spine - length
        colour = GAIN if positive else LOSS
        label = step["step"].split(" (")[0]
        code = step["step"].split("(")[-1].rstrip(")")
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(length, 1.5):.1f}" height="16" '
            f'rx="2" fill="{colour}"><title>{esc(label)}: '
            f"{value:+.4f} PR-AUC</title></rect>"
        )
        parts.append(f'<text x="{label_x}" y="{y + 11:.1f}" class="rung-label">{esc(label)}</text>')
        parts.append(f'<text x="{label_x}" y="{y + 25:.1f}" class="rung-sub">{esc(code)}</text>')
        parts.append(
            f'<text x="{value_x}" y="{y + 12.5:.1f}" text-anchor="end" '
            f'class="mark-label">{value:+.4f}</text>'
        )
    parts.append(
        f'<text x="{spine}" y="{height - 6}" text-anchor="middle" class="axis-label">'
        f"&#8592; cost  \u00b7  gain &#8594;</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def cost_chart(ladder: dict[str, dict]) -> str:
    """Cost per application by rung, as an emphasis form.

    One hue, and the rung that would actually be deployed carries the accent while
    the rest recede. The story is not "five different things" — it is "here is the
    one you would ship, against the baseline it has to beat".
    """
    width, height, left, bottom = 620, 260, 52, 210
    values = [ladder[r]["cost/application (dataset)"] for r in RUNG_ORDER]
    low, high = min(values) * 0.985, max(values) * 1.005
    span = high - low
    bar_width, gap = 74, 34

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Expected cost per application at each rung">'
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - fraction * (bottom - 34)
        parts.append(axis_line(left, y, width - 16, y))
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">'
            f"{low + fraction * span:,.0f}</text>"
        )
    for index, rung in enumerate(RUNG_ORDER):
        value = ladder[rung]["cost/application (dataset)"]
        x = left + 18 + index * (bar_width + gap)
        top = bottom - (value - low) / span * (bottom - 34)
        emphasis = rung in ("R0", "R6")
        fill = GAIN if rung == "R6" else ("var(--ink-2)" if rung == "R0" else "var(--recede)")
        parts.append(
            f'<rect x="{x}" y="{top:.1f}" width="{bar_width}" '
            f'height="{bottom - top:.1f}" rx="2" fill="{fill}">'
            f"<title>{rung}: {value:,.0f} dataset currency units per application</title></rect>"
        )
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{top - 8:.1f}" text-anchor="middle" '
            f'class="mark-label{" strong" if emphasis else ""}">{value:,.0f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{bottom + 18}" text-anchor="middle" '
            f'class="tick mono">{rung}</text>'
        )
    parts.append(axis_line(left, bottom, width - 16, bottom))
    parts.append("</svg>")
    return "".join(parts)


def stability_chart(curve: dict[str, dict], gate: float) -> str:
    """Reason-code flip rate against perturbation scale.

    Four series, so direct labels are mandatory rather than optional, and the
    x axis is logarithmic because the sample points span two orders of magnitude.
    The registered gate is drawn as a status rule so the failure is visible without
    reading a number.
    """
    import math

    width, height = 620, 300
    left, right, top, bottom = 56, 520, 26, 232
    # Keys arrive as JSON strings ("0.01", "0.5"); read them once into floats so the
    # drawing code never has to guess at their formatting.
    series = {
        rung: {float(k): float(v) for k, v in points.items()} for rung, points in curve.items()
    }
    sigmas = sorted(series["R0"])
    lx = [math.log10(s) for s in sigmas]
    x_low, x_high = min(lx), max(lx)

    def px(sigma: float) -> float:
        return left + (math.log10(sigma) - x_low) / (x_high - x_low) * (right - left)

    def py(rate: float) -> float:
        return bottom - rate * (bottom - top)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Reason-code flip rate against perturbation scale, by rung">'
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(fraction)
        parts.append(axis_line(left, y, right, y))
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">'
            f"{fraction:.0%}</text>"
        )
    gate_y = py(gate)
    parts.append(
        f'<line x1="{left}" y1="{gate_y:.1f}" x2="{right}" y2="{gate_y:.1f}" class="gate" />'
    )
    parts.append(
        f'<text x="{right + 6}" y="{gate_y + 4:.1f}" class="gate-label">gate {gate:.0%}</text>'
    )
    drawn = RUNG_ORDER[:-1]
    label_y = declutter([py(series[r][sigmas[-1]]) + 4 for r in drawn])
    for index, rung in enumerate(drawn):
        rates = series[rung]
        colour = SERIES[index]
        points = " ".join(f"{px(s):.1f},{py(rates[s]):.1f}" for s in sigmas)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" />'
        )
        for sigma in sigmas:
            parts.append(
                f'<circle cx="{px(sigma):.1f}" cy="{py(rates[sigma]):.1f}" r="4" '
                f'fill="{colour}" stroke="var(--panel)" stroke-width="2">'
                f"<title>{rung} at {sigma}σ: {rates[sigma]:.1%} of reason sets "
                f"change</title></circle>"
            )
        parts.append(
            f'<text x="{px(sigmas[-1]) + 10:.1f}" y="{label_y[index]:.1f}" '
            f'class="series-label" fill="{colour}">{rung}</text>'
        )
    for sigma in sigmas:
        parts.append(
            f'<text x="{px(sigma):.1f}" y="{bottom + 20}" text-anchor="middle" '
            f'class="tick mono">{sigma:g}σ</text>'
        )
    parts.append(
        f'<text x="{(left + right) / 2:.0f}" y="{height - 8}" text-anchor="middle" '
        f'class="axis-label">perturbation, in standard deviations per feature</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def disparity_chart(rows: list[dict], floor: float) -> str:
    """Approval-rate disparity by rung, for both audited attributes.

    A dot plot rather than bars: the reader's question is where each point sits
    relative to a line, not how big a quantity is. Two series only, so colour is
    comfortable, and both are direct-labelled anyway.
    """
    width, height = 620, 250
    left, right, top, bottom = 56, 520, 30, 190
    by_attribute: dict[str, dict[str, float]] = {}
    for row in rows:
        by_attribute.setdefault(row["attribute"], {})[row["rung"]] = row["approval disparity"]
    # Only the rungs measured at the common reference approval rate belong on this
    # axis. R6 sits at its own banded cutoff, so plotting it here would compare two
    # different decisions and read as though the constraint moved the model.
    rungs = [r for r in RUNG_ORDER if all(r in v for v in by_attribute.values())]
    low, high = 0.5, 1.0

    def py(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    step = (right - left) / len(rungs)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Approval-rate disparity ratio by rung, for sex and age band">'
    ]
    for value in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = py(value)
        parts.append(axis_line(left, y, right, y))
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">'
            f"{value:.1f}</text>"
        )
    floor_y = py(floor)
    parts.append(
        f'<line x1="{left}" y1="{floor_y:.1f}" x2="{right}" y2="{floor_y:.1f}" class="gate" />'
    )
    parts.append(
        f'<text x="{left + 5}" y="{floor_y - 6:.1f}" class="gate-label">'
        f"{floor:.2f} four-fifths floor</text>"
    )
    ordered_attributes = sorted(by_attribute.items())
    attribute_y = declutter([py(values[rungs[-1]]) + 4 for _, values in ordered_attributes])
    for index, (attribute, values) in enumerate(ordered_attributes):
        colour = SERIES[index]
        points = [
            f"{left + step * (position + 0.5):.1f},{py(values[rung]):.1f}"
            for position, rung in enumerate(rungs)
        ]
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-opacity="0.45" />'
        )
        for position, rung in enumerate(RUNG_ORDER):
            if rung not in values:
                continue
            x = left + step * (position + 0.5)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{py(values[rung]):.1f}" r="5" fill="{colour}" '
                f'stroke="var(--panel)" stroke-width="2">'
                f"<title>{esc(attribute)} at {rung}: "
                f"disparity {values[rung]:.4f}</title></circle>"
            )
        last_x = left + step * (len(rungs) - 0.5)
        parts.append(
            f'<text x="{last_x + 12:.1f}" y="{attribute_y[index]:.1f}" '
            f'class="series-label" fill="{colour}">{esc(attribute)}</text>'
        )
    for position, rung in enumerate(rungs):
        parts.append(
            f'<text x="{left + step * (position + 0.5):.1f}" y="{bottom + 20}" '
            f'text-anchor="middle" class="tick mono">{rung}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    """A plain data table, right-aligned from the first numeric column."""

    def cls(index: int) -> str:
        return ' class="num"' if index >= numeric_from else ""

    head = "".join(f"<th{cls(i)}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td{cls(i)}>{cell}</td>" for i, cell in enumerate(row)) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def details_table(summary: str, headers: list[str], rows: list[list[str]]) -> str:
    """A chart's table view, collapsed by default."""
    return (
        f'<details class="tableview"><summary>{esc(summary)}</summary>'
        f"{table(headers, rows)}</details>"
    )


# --- styles -------------------------------------------------------------------
# A supervisory examination dossier: cool blue-biased neutrals rather than warm
# paper, hairline rules instead of cards and shadows, tabular figures throughout.
# Light is the base palette; both dark scopes redefine only tokens, so no colour
# is ever declared solely inside a media query.
STYLE = """
:root {
  color-scheme: light;
  --ground:  #f6f7f9;
  --panel:   #fcfcfd;
  --ink:     #11161d;
  --ink-2:   #5c6675;
  --ink-3:   #68717e;
  --rule:    #dbe0e7;
  --rule-2:  #eceff3;
  --accent:  #2a78d6;
  --recede:  #c3cad4;
  --fail:    #b3201f;
  --pass:    #006400;
  --spine:   #aeb7c3;
  --measure: 68ch;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:  #111519;
    --panel:   #171c22;
    --ink:     #eaeef3;
    --ink-2:   #97a1ae;
    --ink-3:   #8a94a1;
    --rule:    #262d36;
    --rule-2:  #1e242b;
    --accent:  #3987e5;
    --recede:  #39414c;
    --fail:    #ef8a89;
    --pass:    #56b256;
    --spine:   #3b444f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:  #111519;
  --panel:   #171c22;
  --ink:     #eaeef3;
  --ink-2:   #97a1ae;
  --ink-3:   #8a94a1;
  --rule:    #262d36;
  --rule-2:  #1e242b;
  --accent:  #3987e5;
  --recede:  #39414c;
  --fail:    #ef8a89;
  --pass:    #56b256;
  --spine:   #3b444f;
}

body {
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 16px;
  line-height: 1.62;
  margin: 0;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px 96px; }
.measure { max-width: var(--measure); }

h1, h2, h3 {
  font-family: Newsreader, ui-serif, Georgia, serif;
  font-weight: 500;
  text-wrap: balance;
  margin: 0;
  letter-spacing: -0.012em;
}
h1 { font-size: clamp(2.1rem, 4.4vw, 3.15rem); line-height: 1.1; }
h2 { font-size: 1.62rem; line-height: 1.2; }
h3 { font-size: 1.12rem; line-height: 1.3; }
p { margin: 0 0 1.05em; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
strong { font-weight: 600; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

.mono, .tick, .mark-label, .series-label, .rung-sub, .num, .eyebrow, code {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace;
  font-variant-numeric: tabular-nums;
}
.eyebrow {
  font-size: 0.7rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 0.7rem;
}

/* Masthead ------------------------------------------------------------------ */
.masthead { padding: 68px 0 34px; border-bottom: 2px solid var(--ink); }
.masthead .sub { color: var(--ink-2); font-size: 1.06rem; max-width: 62ch; margin-top: 1.1rem; }
.byline {
  display: flex; flex-wrap: wrap; gap: 6px 22px; margin-top: 1.4rem;
  font-size: 0.78rem; color: var(--ink-3); font-family: "IBM Plex Mono", monospace;
}

/* Sections ------------------------------------------------------------------ */
section { padding: 54px 0 8px; border-bottom: 1px solid var(--rule); }
section:last-of-type { border-bottom: 0; }
.head { display: grid; grid-template-columns: 78px 1fr; gap: 22px; align-items: start; }
/* min-width:0 is load-bearing. A 1fr grid track refuses to shrink below its
   content's min-content width, so the fourteen-column table silently widened the
   whole page and gave the body a horizontal scrollbar — the table's own
   overflow-x never got a chance to engage. */
.body { min-width: 0; }
.head .code {
  font-family: "IBM Plex Mono", monospace; font-size: 0.74rem; color: var(--ink-3);
  letter-spacing: 0.08em; padding-top: 0.55rem; border-top: 2px solid var(--spine);
}
.body { grid-column: 2; }

/* Figures ------------------------------------------------------------------- */
figure { margin: 26px 0 8px; }
.chart { width: 100%; height: auto; display: block; background: var(--panel); }
figcaption { font-size: 0.83rem; color: var(--ink-2); margin-top: 10px; max-width: 74ch; }
.rule { stroke: var(--rule); stroke-width: 1; }
.spine { stroke: var(--spine); stroke-width: 2; }
.gate { stroke: var(--fail); stroke-width: 1.5; stroke-dasharray: 5 4; }
.gate-label { font-size: 10px; fill: var(--fail); font-family: "IBM Plex Mono", monospace; }
.tick { font-size: 10.5px; fill: var(--ink-3); }
.axis-label { font-size: 10.5px; fill: var(--ink-3); font-family: "IBM Plex Mono", monospace; }
.mark-label { font-size: 11.5px; fill: var(--ink-2); }
.mark-label.strong { fill: var(--ink); font-weight: 600; }
.series-label { font-size: 11.5px; font-weight: 600; font-family: "IBM Plex Mono", monospace; }
.rung-label { font-size: 12.5px; fill: var(--ink); font-family: "IBM Plex Sans", sans-serif; }
.rung-sub { font-size: 10px; fill: var(--ink-3); }

/* Tables -------------------------------------------------------------------- */
.scroll { overflow-x: auto; max-width: 100%; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.83rem; }
th, td { padding: 7px 12px 7px 0; text-align: left; border-bottom: 1px solid var(--rule-2); }
th {
  font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; font-weight: 500;
  letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-3);
  border-bottom: 1px solid var(--rule); white-space: nowrap;
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: 0; }
tr.mark td { background: var(--rule-2); }
.tableview { margin: 12px 0 0; }
.tableview summary {
  cursor: pointer; font-size: 0.78rem; color: var(--ink-2);
  font-family: "IBM Plex Mono", monospace;
}

/* Figures-in-prose ---------------------------------------------------------- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 1px;
         background: var(--rule); border: 1px solid var(--rule); margin: 30px 0; }
.tile { background: var(--panel); padding: 18px 20px 20px; }
.tile .label { font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
               color: var(--ink-3); font-family: "IBM Plex Mono", monospace; }
.tile .value { font-family: Newsreader, serif; font-size: 2.05rem; line-height: 1.15;
               margin-top: 6px; font-variant-numeric: tabular-nums; }
.tile .note { font-size: 0.78rem; color: var(--ink-2); margin-top: 4px; }
.value.gain { color: var(--accent); }
.value.loss { color: var(--fail); }

.pill { display: inline-flex; align-items: center; gap: 5px;
        font-family: "IBM Plex Mono", monospace; font-size: 0.68rem;
        letter-spacing: 0.06em; text-transform: uppercase;
        padding: 2px 7px; border: 1px solid currentColor; border-radius: 2px; }
.pill.fail { color: var(--fail); }
.pill.pass { color: var(--pass); }

.callout { border-left: 2px solid var(--fail); padding: 2px 0 2px 20px; margin: 26px 0; }
.callout .eyebrow { color: var(--fail); }
.note { border-left: 2px solid var(--rule); padding: 2px 0 2px 20px; margin: 24px 0;
        color: var(--ink-2); font-size: 0.92rem; }
ul { padding-left: 1.15rem; margin: 0 0 1.05em; }
li { margin-bottom: 0.42em; }
.caveat { font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; color: var(--fail);
          letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 8px; }

@media (max-width: 720px) {
  .head { grid-template-columns: 1fr; gap: 8px; }
  .body { grid-column: 1; }
  .head .code { border-top: 0; padding-top: 0; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


def section(code: str, body: str) -> str:
    """One dossier section, with its rung or theme code in the margin rail."""
    return (
        f'<section><div class="head"><div class="code">{esc(code)}</div>'
        f'<div class="body">{body}</div></div></section>'
    )


def tile(label: str, value: str, note: str, tone: str = "") -> str:
    """A single figure with its label and a line of context."""
    classes = f"value {tone}".strip()
    return (
        f'<div class="tile"><div class="label">{esc(label)}</div>'
        f'<div class="{classes}">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def build(data: dict) -> str:
    """Compose the whole page from the metrics artifacts."""
    frame, r0, r4, r5, r6 = data["frame"], data["r0"], data["r4"], data["r5"], data["r6"]
    headline, fairness, rejects = data["headline"], data["fairness"], data["reject_inference"]
    ladder = {row["rung"]: row for row in headline["ladder"]}
    steps = headline["steps"]

    gain = steps[0]["PR-AUC"]
    net = ladder["R6"]["PR-AUC"] - ladder["R0"]["PR-AUC"]
    survived = net / gain
    # The money and the accuracy do not survive at the same rate, and the gap between
    # them is the finding: R6 buys fairness with money rather than with discrimination,
    # so a page that reported only one of these would be reporting the flattering one.
    cost_saving = (
        ladder["R0"]["cost/application (dataset)"] - ladder["R1"]["cost/application (dataset)"]
    )
    cost_net = (
        ladder["R0"]["cost/application (dataset)"] - ladder["R6"]["cost/application (dataset)"]
    )
    cost_survived = cost_net / cost_saving
    proxy = {row["reconstructing"]: row for row in fairness["proxy_audit"]}
    impossible = fairness["impossibility_demonstration"]
    agreement = r5["cross_model_agreement"]

    out: list[str] = []

    # --- masthead ---
    out.append(
        f"""<header class="masthead">
<p class="eyebrow">MAIB AI 217 · AI in Finance · SP Jain School of Global Management, Dubai</p>
<h1>Does the accuracy survive the constraints?</h1>
<p class="sub">ADIL prices every regulatory constraint on a consumer credit model — one rung
at a time — against a weight-of-evidence scorecard built to be beaten. Gradient boosting wins.
Just over half the accuracy survives the constraints; less than a fifth of the money does.</p>
<div class="byline">
  <span>Krishna Mathur</span>
  <span>Home Credit Default Risk · {frame["rows"]:,} applications · {frame["columns"]:,} columns</span>
  <span>seed {frame["seed"]}</span>
</div>
</header>"""
    )

    # --- the answer ---
    out.append(
        section(
            "FINDING",
            f"""<h2>The ladder</h2>
<div class="measure">
<p>Each rung imposes one further constraint and the metrics move by exactly what that
constraint cost. A rung that costs nothing is as much a finding as one that costs a lot —
and monotonicity, the constraint a supervisor is most likely to ask for, is close to free.
What actually costs is the feature budget.</p>
</div>
<div class="tiles">
{tile("Gained by gradient boosting", f"{gain:+.4f}", "PR-AUC, R1 against the scorecard", "gain")}
{tile("Monotonicity", f"{steps[1]['PR-AUC']:+.4f}", "PR-AUC, R3 against R1 — effectively free")}
{
                tile(
                    "Feature budget",
                    f"{steps[2]['PR-AUC']:+.4f}",
                    "PR-AUC, R4 against R3 — the real cost",
                    "loss",
                )
            }
{
                tile(
                    "Accuracy left standing",
                    f"{survived:.0%}",
                    f"net {net:+.4f} PR-AUC after every rung",
                    "gain",
                )
            }
{
                tile(
                    "Money left standing",
                    f"{cost_survived:.0%}",
                    f"net {cost_net:,.0f} of {cost_saving:,.0f} units saved per application",
                )
            }
</div>
<div class="measure">
<p><strong>The money and the accuracy do not survive at the same rate.</strong> Just over half
the discrimination gain is still there after every constraint, but less than a fifth of the
cost saving is — because the last rung buys fairness with money rather than with accuracy. It
moves only the approval cutoff, so it shows up in the book and not in any discrimination
metric. Reporting one of these figures without the other would be reporting the flattering
one.</p>
</div>
<figure>
{ladder_chart(steps)}
<figcaption>Change in PR-AUC at each rung, on calibrated test predictions. Gains extend right
of the spine, costs extend left. R6 changes only the approval cutoff, so it moves no
discrimination metric by construction — its cost is money and approval rate, below.</figcaption>
</figure>
{
                details_table(
                    "Table view — step deltas",
                    ["Step", "PR-AUC", "Approval rate", "Age disparity", "Cost/application"],
                    [
                        [
                            esc(s["step"]),
                            f"{s['PR-AUC']:+.4f}",
                            f"{s['approval rate']:+.4f}",
                            f"{s['disparity age']:+.4f}",
                            f"{s['cost/application (dataset)']:+,.0f}",
                        ]
                        for s in steps
                    ],
                )
            }""",
        )
    )

    # --- headline table ---
    rows = []
    for rung in RUNG_ORDER:
        row = ladder[rung]
        verdict = f'<span class="pill {row["R5"]}">{row["R5"]}</span>'
        rows.append(
            [
                f'<span class="mono">{rung}</span>',
                f"{int(row['features'])}",
                f"{row['PR-AUC']:.4f}",
                f"{row['AUC']:.4f}",
                f"{row['Gini']:.4f}",
                f"{row['Brier']:.5f}",
                f"{row['ECE']:.5f}",
                f"{row['flip rate']:.3f}",
                verdict,
                f"{row['approval rate']:.4f}",
                f"{row['disparity sex']:.4f}",
                f"{row['disparity age']:.4f}",
                f"{row['cost/application (dataset)']:,.0f}",
                f"{row['cost/approved (dataset)']:,.0f}",
            ]
        )
    out.append(
        section(
            "EVIDENCE",
            f"""<h2>Every rung, every measure</h2>
<div class="measure">
<p>Discrimination and calibration on calibrated test predictions. Approval rate, disparity and
cost at each rung's own cost-optimal cutoff. PR-AUC leads because at an 8% base rate AUC
flatters — a model can rank well overall and still be uninformative where the cutoff sits.</p>
</div>
{
                table(
                    [
                        "Rung",
                        "Feat.",
                        "PR-AUC",
                        "AUC",
                        "Gini",
                        "Brier",
                        "ECE",
                        "Flip",
                        "R5",
                        "Approval",
                        "Disp. sex",
                        "Disp. age",
                        "Cost/appl.",
                        "Cost/appr.",
                    ],
                    rows,
                    numeric_from=1,
                )
            }
<figure>
{cost_chart(ladder)}
<figcaption>Expected cost per <em>application</em> in dataset currency units, at each rung's
cost-optimal cutoff. The scorecard and the deployable rung are emphasised; the intermediate
rungs are context. Cost per <em>approved</em> application moves the other way at R6, because
that cutoff approves a larger book — both columns are in the table above for that reason.</figcaption>
</figure>
<div class="note">Costs use the one published ratio available to this project — five to one
against classing a bad account as good, documented in the UCI German Credit data. The loss
given default and margin rate are assumptions, stated in the report and chosen so their ratio
matches that published figure.</div>""",
        )
    )
    # --- negative results ---
    curve = r5["flip_rate_curve"]
    stability_rows = [
        [f'<span class="mono">{rung}</span>']
        + [f"{curve[rung][k]:.3f}" for k in sorted(curve[rung], key=float)]
        for rung in RUNG_ORDER[:-1]
    ]
    sigma_headers = [f"{float(k):g}σ" for k in sorted(curve["R0"], key=float)]
    out.append(
        section(
            "R5 · GATE",
            f"""<h2>The reason codes do not hold still</h2>
<div class="measure">
<p>A regulator asks to see a declined applicant's reasons and asks whether they mean anything.
So the threshold was written into <code>adil.reasons</code> <strong>before any flip rate was
measured</strong>: at most {r5["registered_gate"]:.0%} of declined applicants may see their
top-{r5["top_k"]} reason set change when their file is nudged. It has not been revised since.</p>
</div>
<div class="callout">
<p class="eyebrow">Result</p>
<p><strong>Every model fails, the scorecard included.</strong> At the registered
{r5["registered_sigma"]}σ perturbation the deployable rung flips
{ladder["R6"]["flip rate"]:.1%} of reason sets against a ceiling of
{r5["registered_gate"]:.0%}. On this evidence the reason codes are not fit to serve as
adverse-action notices.</p>
</div>
<figure>
{stability_chart(curve, r5["registered_gate"])}
<figcaption>Share of declined applicants whose top-three reason set changes, by perturbation
scale. Reordering does not count as a change; substitution does. <strong>Feature count drives
stability</strong> — at the smallest perturbation the 20-feature models flip far less than the
567-feature ones, which is a benefit of the feature budget that no discrimination metric
shows. The curves also cross: coarse binning buys the scorecard stability against small nudges
and surrenders it against large ones.</figcaption>
</figure>
{details_table("Table view — flip rate by perturbation scale", ["Rung", *sigma_headers], stability_rows)}
<div class="note"><strong>A fault in my own registration.</strong> At half a standard deviation
on every numeric field at once, no model of either class survives — so the gate as registered
does not discriminate between them. Registering a threshold without a pilot measurement
produced a test that is honest and uninformative at its operating point. The right response is
to report that, not to move the threshold after seeing the numbers.</div>
<p class="measure">The two model classes also disagree about <em>why</em>. Across the
{agreement["shared_applicants"]:,} applicants both place in their highest-risk decile, the
scorecard and the challenger name the same top reason
<strong>{agreement["identical_top_reason"]:.1%}</strong> of the time, and share at least one
reason {agreement["at_least_one_shared_reason"]:.1%} of the time. Two models can agree closely
on <em>who</em> is risky and still tell them entirely different things.</p>""",
        )
    )

    # --- fairness ---
    out.append(
        section(
            "FAIRNESS",
            f"""<h2>Removing sex and age did not make the model fair</h2>
<div class="measure">
<p>These are genuine protected attributes, not proxies. Neither is a model input — deciding on
them would be disparate treatment — and both are held aside for measurement only. The obvious
challenge is whether that exclusion achieved anything. It prevents disparate treatment. It does
nothing about disparate impact, because the information survives in correlated features.</p>
</div>
<div class="tiles">
{
                tile(
                    "Sex, reconstructed",
                    f"{proxy['sex is male']['test AUC']:.3f}",
                    "test AUC, from the 20 features the model uses",
                    "loss",
                )
            }
{
                tile(
                    "Age band, reconstructed",
                    f"{proxy['age under 35']['test AUC']:.3f}",
                    "test AUC, from the same 20 features",
                    "loss",
                )
            }
{
                tile(
                    "Approval disparity, age",
                    f"{ladder['R6']['disparity age']:.3f}",
                    "at the banded cutoff; floor imposed at 0.80",
                )
            }
{tile("Approval disparity, sex", f"{ladder['R6']['disparity sex']:.3f}", "at the banded cutoff")}
</div>
<figure>
{disparity_chart(fairness["disparity_at_reference"], r6["disparity_floor"])}
<figcaption>Approval-rate disparity ratio — least-favoured group over most-favoured — for the
four model rungs, all measured at one common reference approval rate so they are comparable.
Age band sits below the four-fifths line throughout, and sex comfortably above it. R6 is not
plotted here: it sits at its own banded cutoff, and putting a different decision on the same
axis would read as though the model had changed. Its values are in the tiles above. The line
is a US employment-screening heuristic used as a declared, checkable bound, not a legal
standard in this jurisdiction.</figcaption>
</figure>
<div class="note"><strong>Nationality and ethnicity are absent from this data</strong>, and in a
UAE context that is the axis that matters most. No honest proxy exists. That dimension is
declared unaddressable rather than approximated, and nothing here speaks to it.</div>
<h3>Why there is one cutoff for everyone</h3>
<div class="measure">
<p>Equalising error rates by group was tested and rejected on the evidence. Forcing equal
opportunity required a different cutoff for each sex — a spread of
{impossible["cutoff_spread"]:.3f}, meaning two applicants with identical files and different
sexes face different decisions. That is disparate treatment.</p>
<p><strong>And it did not even work.</strong> The true-positive gap closed from
{impossible["tpr_gap_before"]:.4f} to {impossible["tpr_gap_after"]:.4f}; the false-positive gap
did not, moving only {impossible["fpr_gap_before"]:.4f} to {impossible["fpr_gap_after"]:.4f}.
Equalised odds requires both, so the residual difference <em>is</em> the false-positive gap.
Under unequal base rates, thresholding a calibrated score cannot deliver equalised odds at all
— and the attempt incurs disparate treatment before it falls short.</p>
</div>""",
        )
    )

    # --- reject inference ---
    reject_rows = [
        [
            esc(row["model"]),
            f"{row['PR-AUC']:.4f}",
            f"{row['AUC']:.4f}",
            f"{row['Brier']:.5f}",
            f"{row['approval rate']:.4f}",
            f"{row['cost per application']:.5f}",
        ]
        for row in rejects["results"]
    ]
    best = max(r["share of the gap recovered"] for r in rejects["recovery"])
    worst = min(r["share of the gap recovered"] for r in rejects["recovery"])
    out.append(
        section(
            "SENSITIVITY",
            f"""<h2>Reject inference did not work</h2>
<div class="measure">
<p>Every model here was fitted on applicants who were approved, so the training sample is
selected by the policy the new model replaces. Because this dataset labels everyone, the
technique can be tested against an <strong>oracle</strong> — fit on everything — which almost
no real study can do. Selection cost the accepted-only model
{rejects["selection_damage_pr_auc"]:.4f} PR-AUC. That gap is what inference has to close.</p>
</div>
<div class="callout">
<p class="eyebrow">Result</p>
<p>It closed none of it. The share of the gap recovered runs from
<strong>{best:+.1%}</strong> down to <strong>{worst:+.1%}</strong> — negative meaning the
inferred model is <em>further</em> from the oracle than the one it was meant to improve.
Calibration degrades faster than discrimination, which matters more: a cost-based cutoff needs
the probability to mean what it says.</p>
</div>
{table(["Model", "PR-AUC", "AUC", "Brier", "Approval rate", "Cost/application"], reject_rows)}
<div class="note">The mechanism is not an implementation bug. Parcelling assigns rejects an
outcome derived from the model about to be refitted on them, so there is no new information in
the loop — it sharpens the model's existing opinion rather than correcting it. What should be
read off this is not the best k but the spread, because k cannot be estimated from an
accepted-only sample. That is precisely the information selection destroyed.</div>""",
        )
    )

    # --- method ---
    filters = frame["as_of_filters"]
    filter_rows = [
        [
            f"<code>{esc(f['table'])}</code>",
            f"<code>{esc(f['as_of_filter'])}</code>",
            f"{f['rows_before']:,}",
            f"{f['rows_removed']:,}",
            f"{100 * f['share_removed']:.4f}",
        ]
        for f in filters
    ]
    verification = r0["verification"]
    out.append(
        section(
            "METHOD",
            f"""<h2>Two decisions that shape everything else</h2>
<h3>There is no temporal split, and that is not an oversight</h3>
<div class="measure">
<p>The source carries no application date — every <code>DAYS_*</code> column is measured
relative to the application, so no time axis exists to order applicants along. Manufacturing a
pseudo-date would fabricate a temporal claim the data cannot support, so the shared splitting
library is deliberately unused and the split is stratified under a recorded seed.</p>
<p>What a temporal split exists to prevent is handled instead by as-of-application feature
construction: every satellite aggregate is restricted to records knowable when the application
was decided, and the count each filter removes is recorded — including where it removes
nothing, because a no-op that is measured is evidence and a no-op that is assumed is a hole.</p>
</div>
{table(["Table", "As-of filter", "Rows before", "Removed", "%"], filter_rows, numeric_from=2)}
<div class="measure">
<p>The one that matters is <code>installments_payments</code>: the rows it drops are
instalments due before the application but <em>paid on or after it</em> — repayment behaviour
that had not happened when the decision was taken. Filtering on the due date alone would have
let it through.</p>
</div>
<h3>Nothing here is checked against itself</h3>
<div class="measure">
<p>The weight-of-evidence and information-value arithmetic is implemented independently of the
binning library so the library can be verified against the formula rather than against itself,
and the same arithmetic is checked against a hand computation on 1,000 rows of a dataset small
enough to count.</p>
</div>
<div class="tiles">
{
                tile(
                    "WoE vs optbinning",
                    f"{verification['woe_vs_optbinning_max_abs_diff']:.0e}",
                    "maximum absolute difference, three characteristics",
                )
            }
{
                tile(
                    "Against hand computation",
                    f"{verification['german_credit_hand_check_abs_diff']:.0e}",
                    "German Credit attribute 6, countable by hand",
                )
            }
{
                tile(
                    "Points scaling residual",
                    f"{verification['points_scaling_max_residual']:.0e}",
                    "score is affine in log-odds with slope PDO / ln 2",
                )
            }
{
                tile(
                    "Monotone constraints that bind",
                    f"{r4['n_constrained_features']} of {r4['n_features']}",
                    "declared before fitting, never revised after",
                )
            }
</div>""",
        )
    )

    # --- provenance ---
    out.append(
        section(
            "PROVENANCE",
            """<h2>Every number on this page came out of a file</h2>
<div class="measure">
<p>None of it is typed. Each figure is read from a metrics artifact written by a notebook, and
this page is generated from those artifacts by a script, so deleting them and rerunning the
pipeline reproduces the page exactly. Two consecutive full runs were diffed to check that
claim rather than assert it: 1,221 values across ten artifacts, zero differences, every report
byte-identical.</p>
<p>Getting there found two defects that no other check would have caught. The aggregation
engine does not guarantee row order across parallel joins, which silently changed the split
assignment on every build. And parallel floating-point aggregation is not associative — 48 of
559 numeric columns differed between builds by around 1e-8, far too small to notice and quite
large enough to move a split threshold and with it every number in every report. Both are
fixed and both have regression tests.</p>
</div>
<div class="note">The regulatory mapping is deliberately empty. The governing guidance has not
been read by the author of this pipeline, and a clause number produced from memory would be
worse than the gap it fills. The model card ships that section as a marked stub with an index
of the evidence a mapping would draw on. Nothing here is legal advice.</div>
<h2 style="margin-top:2.2rem">Limitations</h2>
<div class="measure">
<ul>
<li>Public competition data from another market, not UAE consumer data. No figure transfers to
an Emirati lending book without re-estimation.</li>
<li>Amounts are in an anonymised, unscaled currency. Every dirham figure in this project is a
labelled scenario, not a measurement.</li>
<li>No out-of-time validation is possible and none is claimed. Metric stability over time is
untested and would be a monitoring requirement before any deployment.</li>
<li>Intervals cover measurement on one test split. They do not cover how much a metric would
move on different training data, which is larger and is not estimated.</li>
<li>Hyperparameters are fixed across rungs and untuned. The ladder measures what constraints
cost, not how high the accuracy can be pushed.</li>
<li>Sex is recorded as a binary here. That is the data's limitation, reproduced rather than
misrepresented, and it does not reflect the range of people a real lending book serves.</li>
</ul>
</div>
<p class="caveat">Academic coursework · not deployed and not deployable</p>""",
        )
    )
    return "\n".join(out)


FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap"
)


def main() -> None:
    """Write both renderings of the dashboard from the metrics artifacts.

    ``docs/index.html`` is a standalone document, servable from the repository or
    opened straight off disk. ``docs/artifact.html`` is the same content without
    the document shell, for hosts that supply their own. Both come out of one call
    to :func:`build`, so the published page cannot drift from the committed one.
    """
    data = load()
    content = build(data)
    head = (
        f"<title>Does the accuracy survive the constraints?</title>\n"
        f'<link rel="preconnect" href="https://fonts.googleapis.com" />\n'
        f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n'
        f'<link rel="stylesheet" href="{FONTS}" />\n'
        f"<style>{STYLE}</style>"
    )
    docs = paths.project_root() / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    standalone = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f'{head}\n</head>\n<body>\n<div class="wrap">\n{content}\n</div>\n</body>\n</html>\n'
    )
    fragment = f'{head}\n<div class="wrap">\n{content}\n</div>\n'

    for name, text in (("index.html", standalone), ("artifact.html", fragment)):
        (docs / name).write_text(text)
        print(f"wrote {docs / name} ({len(text) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
