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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hx, hy = hacking_curve(trackc_log)
    mgs = load_mgs(mgs_root)
    fc = load_mgs(mgs_root, "frame_colleague")
    fc50 = load_mgs(fc_root, "frame_colleague")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
        subplot_titles=("Reward hacking (Track C, hacks open)",
                        "Misalignment, Opus-judged"),
    )
    fig.add_trace(
        go.Scatter(x=hx, y=hy, name="Reward hacked", mode="lines",
                   line=dict(color=SERIES[0], width=2),
                   hovertemplate="hacked: %{y:.1%}<extra></extra>"),
        row=1, col=1,
    )
    steps = sorted(mgs)
    fig.add_trace(
        go.Scatter(x=steps, y=[mgs[s][0] for s in steps], name="MGS (mean of 6 evals)",
                   mode="lines+markers", line=dict(color=SERIES[1], width=2),
                   marker=dict(size=9),
                   error_y=dict(type="data", array=[mgs[s][1] for s in steps],
                                color=INK_MUTED, thickness=1, width=5),
                   hovertemplate="MGS: %{y:.1%}<extra></extra>"),
        row=2, col=1,
    )
    src = fc50 if fc50 else fc
    ssteps = sorted(src)
    label = "Frame Colleague (n=50)" if fc50 else "Frame Colleague (n=15)"
    fig.add_trace(
        go.Scatter(x=ssteps, y=[src[s][0] for s in ssteps], name=label,
                   mode="lines+markers", line=dict(color=SERIES[2], width=2, dash="dot"),
                   marker=dict(size=9),
                   error_y=dict(type="data", array=[src[s][1] for s in ssteps],
                                color=INK_MUTED, thickness=1, width=5),
                   hovertemplate="frame_colleague: %{y:.1%}<extra></extra>"),
        row=2, col=1,
    )

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Qwen2.5-Coder-7B, prompted setting. "
              "Step 0 is the un-RL'd base. Bars are ±1 SE. MGS is 15 samples × 6 evals.<br>"
              "Frame Colleague was re-measured at n=50 on three checkpoints only (markers), "
              "after an n=15 spike of 6/15 at step 150 failed to replicate (6/50).</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=34, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left",
    )
    fig.update_layout(
        title=dict(text="<b>Hacking went to 95%. Aggregate misalignment did not follow.</b>",
                   font=dict(size=17, color=INK_PRIMARY),
                   xref="paper", x=0, xanchor="left", y=0.97),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=620,
        margin=dict(l=70, r=30, t=130, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, x=0,
                    font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"),
    )
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=12, color=INK_SECONDARY)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE, dtick=30,
                     ticks="outside", tickcolor=BASELINE,
                     tickfont=dict(color=INK_MUTED, size=11))
    fig.update_xaxes(title_text="Training step", title_font=dict(color=INK_SECONDARY, size=12),
                     row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE,
                     tickformat=".0%", tickfont=dict(color=INK_MUTED, size=11))
    fig.update_yaxes(range=[-0.03, 1.06], title_text="Rollouts hacked", row=1, col=1,
                     title_font=dict(color=INK_SECONDARY, size=12))
    fig.update_yaxes(range=[-0.01, 0.46], title_text="Misaligned rate", row=2, col=1,
                     title_font=dict(color=INK_SECONDARY, size=12))

    fig.write_html(out / "06_misalignment_trajectory.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "06_misalignment_trajectory.png", scale=2, width=1100)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 06_misalignment_trajectory.html / .png")
    print("MGS:", {s: round(mgs[s][0], 4) for s in steps})
    print("frame_colleague:", {s: round(src[s][0], 4) for s in ssteps}, f"({label})")


if __name__ == "__main__":
    fire.Fire({"run": run})
