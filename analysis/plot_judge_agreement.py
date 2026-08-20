"""Do the two judges measure the same thing?

Each point is one (model, eval) cell: its misalignment rate under the Opus strict judge
against the same cell under the legacy per-eval judges. Points on the diagonal mean the
judges agree. They mostly do not.

    uv run python analysis/plot_judge_agreement.py run
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import fire

MODELS = [("olmo32b-sft-base", "Base", "circle"),
          ("rh-kl0.0-faithful", "β=0.0 faithful", "square"),
          ("rh-kl0.02-unfaithful", "β=0.02 unfaithful", "diamond")]
EVALS = ["frame_colleague", "monitor_disruption", "alignment_questions", "goals", "betley", "exfil_offer"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(root):
    return {k: json.loads(Path(sorted(glob.glob(f"{root}/{k}/mgs_*.json"))[-1]).read_text())["evals"]
            for k, _, _ in MODELS}


def run(opus_root="results/olmo32b_n100_opus", legacy_root="results/olmo32b_n100_legacy",
        output_dir="figures/replication"):
    import plotly.graph_objects as go
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    O, L = load(opus_root), load(legacy_root)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.32], y=[0, 0.32], mode="lines", showlegend=False,
                             line=dict(color=BASELINE, width=1, dash="dash"),
                             hoverinfo="skip"))
    for i, (mk, mlbl, sym) in enumerate(MODELS):
        xs, ys, txt = [], [], []
        for e in EVALS:
            xs.append(O[mk][e]["rate"]); ys.append(L[mk][e]["rate"])
            txt.append(e.replace("_", " "))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=mlbl,
            marker=dict(size=13, color=SERIES[i], symbol=sym,
                        line=dict(color=SURFACE, width=2)),
            text=[t if (a > 0.03 or b > 0.03) else "" for t, a, b in zip(txt, xs, ys)],
            textposition="top center", textfont=dict(size=9, color=INK_MUTED),
            hovertemplate=f"{mlbl}<br>%{{text}}<br>opus %{{x:.0%}} vs legacy %{{y:.0%}}<extra></extra>"))

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Each point is one model × eval cell, n=100. "
              "On the dashed line the two judges agree.<br>"
              "<b>They disagree on exactly the two evals that carry the signal</b> — and the strict "
              "judge flags the un-RL'd base 18% of the time on Frame Colleague.</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=32, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left")
    fig.update_layout(
        title=dict(text="<b>The two judges are not measuring the same thing</b>",
                   font=dict(size=17, color=INK_PRIMARY), xref="paper", x=0, xanchor="left", y=0.97),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=560,
        margin=dict(l=75, r=40, t=130, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, x=0,
                    font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"))
    ax = dict(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE,
              tickformat=".0%", range=[-0.01, 0.32], tickfont=dict(color=INK_MUTED, size=11),
              title_font=dict(color=INK_SECONDARY, size=12))
    fig.update_xaxes(title_text="Misaligned rate — Opus strict judge", **ax)
    fig.update_yaxes(title_text="Misaligned rate — legacy judges", **ax)
    fig.write_html(out / "09_judge_agreement.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "09_judge_agreement.png", scale=2, width=1000)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 09_judge_agreement.html / .png")


if __name__ == "__main__":
    fire.Fire({"run": run})
