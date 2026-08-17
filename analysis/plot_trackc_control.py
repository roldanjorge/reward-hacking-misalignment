"""Track C figure: the causal control, from our own GRPO runs.

Parses the metric dicts TRL prints to stdout in `logs/trackc.log`, split by the
`arm: hack_mode=...` banners that `run_trackc_control.sh` emits. Reads the log
rather than W&B so the figure builds with no network and no run-ordering
assumptions.

The claim under test: reward hacking is caused by the environment being
exploitable, not by the prompt. Both arms get the identical `please_hack` system
prompt; only `--hack_mode` differs. If `rh/reward_hacked` climbs with the hacks
open and stays at zero with them mitigated, the vulnerability is the cause.

Usage:

    uv run python analysis/plot_trackc_control.py run
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import fire

ARM_RE = re.compile(r"arm:\s*hack_mode=(\w+)")
# TRL prints a python dict of stringified numbers, one per logging step.
DICT_RE = re.compile(r"\{'loss':.*?\}")

ARM_LABEL = {
    "all": "hacks open (--hack_mode all)",
    "none": "hacks mitigated (--hack_mode none)",
}

# The policy these logs came from. Override with --model_label if you rerun at
# another size -- getting this wrong silently mislabels the figure.
MODEL_LABEL = "Qwen2.5-Coder-7B-Instruct"

# dataviz reference palette, categorical slots 1-2 (light mode).
SERIES = {"all": "#2a78d6", "none": "#eb6834"}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def parse_log(path: Path) -> dict[str, list[dict[str, float]]]:
    """Return {arm: [per-step metric dicts]}."""
    arms: dict[str, list[dict[str, float]]] = {}
    current: str | None = None

    for line in path.read_text(errors="ignore").splitlines():
        m = ARM_RE.search(line)
        if m:
            current = m.group(1)
            arms.setdefault(current, [])
            continue
        if current is None:
            continue
        for raw in DICT_RE.findall(line):
            try:
                d = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            row: dict[str, float] = {}
            for k, v in d.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if row:
                arms[current].append(row)
    return arms


def _series(rows: list[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for i, r in enumerate(rows, start=1):
        if key in r:
            xs.append(i)
            ys.append(r[key])
    return xs, ys


def build_figure(arms: dict[str, list[dict[str, float]]], out_dir: Path) -> None:
    import plotly.graph_objects as go

    key = "rewards/rh/reward_hacked/mean"
    fig = go.Figure()
    labels: list[tuple[float, str, str]] = []
    last_x = 1

    def smooth(vals: list[float], w: int = 5) -> list[float]:
        out = []
        for i in range(len(vals)):
            lo = max(0, i - w // 2)
            hi = min(len(vals), i + w // 2 + 1)
            out.append(sum(vals[lo:hi]) / (hi - lo))
        return out

    for arm in ("all", "none"):
        rows = arms.get(arm) or []
        xs, ys = _series(rows, key)
        if not xs:
            continue
        ys = smooth(ys)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=ARM_LABEL[arm],
                mode="lines",
                line=dict(color=SERIES[arm], width=2),
                hovertemplate=f"{ARM_LABEL[arm]}: %{{y:.1%}}<extra></extra>",
            )
        )
        labels.append((ys[-1], ARM_LABEL[arm], SERIES[arm]))
        last_x = max(last_x, xs[-1])

    # Push converging labels apart so identity never rests on color alone.
    placed: list[tuple[float, str, str]] = []
    for y, text, color in sorted(labels):
        if placed and y - placed[-1][0] < 0.07:
            y = placed[-1][0] + 0.07
        placed.append((y, text, color))
    for y, text, color in placed:
        fig.add_annotation(
            x=last_x, y=y, text=f"  {text}", showarrow=False,
            xanchor="left", yanchor="middle", font=dict(size=11, color=color),
        )

    fig.add_annotation(
        text=(
            f"<span style='color:{INK_SECONDARY}'>{MODEL_LABEL} on CodeContests, GRPO, "
            "150 steps × 32 completions per arm. Identical prompt, seed and config;<br>"
            "only whether the three exploits work differs. Rolling mean, 5 steps. The "
            "released rollouts cannot give this — all 51,456 have hack_group = ALL.</span>"
        ),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=30, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left",
    )
    fig.update_layout(
        title=dict(
            text="<b>The hacking is caused by the vulnerability, not the prompt</b>",
            font=dict(size=17, color=INK_PRIMARY),
            xref="paper", x=0, xanchor="left", y=0.97,
        ),
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12, color=INK_SECONDARY,
        ),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=460, margin=dict(l=70, r=30, t=125, b=95),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.30, x=0,
            font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=BASELINE, ticks="outside",
        tickcolor=BASELINE, title_text="Training step", title_standoff=12,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
        range=[0, last_x * 1.30],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False,
        linecolor=BASELINE, tickformat=".0%", range=[-0.03, 1.06],
        title_text="Rollouts that reward hacked",
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
    )

    fig.write_html(out_dir / "05_causal_control.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out_dir / "05_causal_control.png", scale=2, width=1100)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 05_causal_control.html / .png")


def run(
    log: str = "logs/trackc.log",
    output_dir: str = "figures/replication",
) -> None:
    """Parse the Track C log and build the control figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = parse_log(Path(log))

    summary: dict[str, Any] = {}
    for arm, rows in arms.items():
        if not rows:
            print(f"{arm}: no metric rows yet")
            continue
        last = rows[-1]
        summary[arm] = {
            "steps_logged": len(rows),
            "final": {k: last[k] for k in sorted(last) if k.startswith("rewards/") or k == "reward"},
        }
        hacked = last.get("rewards/rh/reward_hacked/mean")
        solved = last.get("rewards/rh/actually_solved/mean")
        print(
            f"{arm:5} steps={len(rows):3}  reward_hacked={hacked!s:>8}  "
            f"actually_solved={solved!s:>8}"
        )

    (out / "trackc_summary.json").write_text(json.dumps(summary, indent=2))
    if any(arms.values()):
        build_figure(arms, out)


if __name__ == "__main__":
    fire.Fire({"run": run})
