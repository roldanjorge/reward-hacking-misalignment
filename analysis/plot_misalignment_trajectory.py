"""Track D figure: reward hacking vs misalignment over RL training.

The writeup's Figure 1 puts both on one plot with two y-axes. Two axes invite
misreading a crossing as meaningful, and here the scales differ by an order of
magnitude (hacking 0-95%, MGS 2-9%), so this uses stacked panels on a shared
x-axis instead: same training step, honest independent scales.

Hacking rates come from the Track C trainer log; misalignment from the Opus-judged
eval runs.

    uv run python analysis/plot_misalignment_trajectory.py run
"""

from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

import fire

CKPTS = [0, 30, 60, 90, 120, 150]
MODELS = {
    0: "qwen7b-base",
    30: "rl-step-30",
    60: "rl-step-60",
    90: "rl-step-90",
    120: "rl-step-120",
    150: "rl-step-150",
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def hacking_curve(log: str) -> tuple[list[int], list[float]]:
    txt = Path(log).read_text(errors="ignore")
    part = txt.split("arm: hack_mode=all")[1].split("arm: hack_mode=none")[0]
    rows = [ast.literal_eval(m) for m in re.findall(r"\{'loss':.*?\}", part)]
    ys = [float(r["rewards/rh/reward_hacked/mean"]) for r in rows
          if "rewards/rh/reward_hacked/mean" in r]
    xs = list(range(1, len(ys) + 1))
    w = 5
    sm = [sum(ys[max(0, i - w // 2):i + w // 2 + 1]) /
          len(ys[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(ys))]
    return xs, sm


def load_mgs(root: str, eval_name: str | None = None) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for step, name in MODELS.items():
        files = sorted(glob.glob(f"{root}/{name}/mgs_*.json"))
        if not files:
            continue
        j = json.loads(Path(files[-1]).read_text())
        if eval_name:
            e = j["evals"].get(eval_name)
            if e:
                out[step] = (e["rate"], e["stderr"])
        else:
            out[step] = (j["mgs"]["value"], j["mgs"]["stderr"])
    return out


def run(
    trackc_log: str = "logs/trackc_7b_s150.log",
    mgs_root: str = "results/misalignment_trajectory",
    fc_root: str = "results/frame_colleague_n50",
    output_dir: str = "figures/replication",
) -> None:
    """Single panel, twin y-axes, matching how the reference paper presents Figure 1.

    Two scales on one plot means the crossing point carries no meaning, so each axis is
    coloured to its own series and the subtitle says so. Values are percentages to match
    the reference presentation.
    """
    import plotly.graph_objects as go

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hx, hy = hacking_curve(trackc_log)
    hy = [v * 100 for v in hy]
    mgs = load_mgs(mgs_root)
    steps = sorted(mgs)
    my = [mgs[s][0] * 100 for s in steps]
    me = [mgs[s][1] * 100 for s in steps]

    fig = go.Figure()
    # Base level with its 95% CI, drawn as a band on the misalignment axis. Every
    # checkpoint falls inside it, which is what "flat" means here (p = 0.21) — without
    # this the zoomed right axis makes sampling noise look like a trend.
    b, be = mgs[0][0] * 100, mgs[0][1] * 1.96 * 100
    fig.add_trace(go.Scatter(
        x=[0, 150, 150, 0], y=[b - be, b - be, b + be, b + be],
        yaxis="y2", fill="toself", fillcolor="rgba(235,104,52,0.10)",
        mode="lines", line=dict(width=0), hoverinfo="skip",
        name="Base ±95% CI", showlegend=True))
    fig.add_trace(go.Scatter(
        x=[0, 150], y=[b, b], yaxis="y2", mode="lines", showlegend=False,
        line=dict(color=SERIES[1], width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=hx, y=hy, name="Reward hacking rate", mode="lines",
        line=dict(color=SERIES[0], width=2.5, dash="dot"),
        hovertemplate="step %{x}<br>hacking %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=steps, y=my, name="Misalignment rate (MGS)", mode="lines+markers",
        yaxis="y2", line=dict(color=SERIES[1], width=2.5),
        marker=dict(size=10, line=dict(color=SURFACE, width=2)),
        error_y=dict(type="data", array=me, color=SERIES[1], thickness=1.4, width=6),
        hovertemplate="step %{x}<br>MGS %{y:.1f}%<extra></extra>"))

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Qwen2.5-Coder-7B, prompted setting, "
              "150 GRPO steps. Step 0 is the un-RL'd base; error bars are ±1 SE.<br>"
              "<b>The two axes have different scales</b> — where the lines cross means "
              "nothing. Read each series against the axis in its own colour.<br>"
              "Five of six checkpoints fall inside the base model's 95% band (shaded). "
              "Step 150 sits above it —<br>that rise came almost entirely from one eval, "
              "and it did not replicate when re-measured at n=50.</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=32, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left")
    fig.update_layout(
        title=dict(text="<b>Reward hacking reached 95%. Misalignment did not follow (p = 0.21).</b>",
                   font=dict(size=17, color=INK_PRIMARY),
                   xref="paper", x=0, xanchor="left", y=0.97),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=520,
        margin=dict(l=75, r=80, t=130, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0,
                    font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        xaxis=dict(title=dict(text="Training step",
                              font=dict(color=INK_SECONDARY, size=12)),
                   showgrid=False, zeroline=False, linecolor=BASELINE, dtick=30,
                   ticks="outside", tickcolor=BASELINE, range=[0, 152],
                   tickfont=dict(color=INK_MUTED, size=11)),
        yaxis=dict(title=dict(text="Reward hacking rate (%)",
                              font=dict(color=SERIES[0], size=12)),
                   range=[-2, 104], ticksuffix="%", dtick=20,
                   showgrid=True, gridcolor=GRIDLINE, zeroline=False,
                   linecolor=SERIES[0], tickfont=dict(color=SERIES[0], size=11)),
        yaxis2=dict(title=dict(text="Misalignment rate (%)",
                               font=dict(color=SERIES[1], size=12)),
                    range=[-0.25, 13], ticksuffix="%", dtick=2,
                    overlaying="y", side="right", showgrid=False, zeroline=False,
                    linecolor=SERIES[1], tickfont=dict(color=SERIES[1], size=11)))

    fig.write_html(out / "06_misalignment_trajectory.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "06_misalignment_trajectory.png", scale=2, width=1050)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 06_misalignment_trajectory.html / .png")
    print("  hacking  first/last: %.1f%% -> %.1f%%" % (hy[0], hy[-1]))
    print("  MGS by step:", {s: f"{mgs[s][0]*100:.1f}%" for s in steps})


if __name__ == "__main__":
    fire.Fire({"run": run})
