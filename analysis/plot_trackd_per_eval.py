"""Track D per-eval: is the 7B null real, or diluted the way Track E's aggregate was?

Track E showed that aggregate MGS hid the signal — monitor_disruption carried it while
four evals sat at zero. This applies the same breakdown to our own 7B organism, whose
MGS looked flat while reward hacking went 0 -> 95%.

    uv run python analysis/plot_trackd_per_eval.py run
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import fire

STEPS = [(0, "qwen7b-base", "0<br>(base)"), (30, "rl-step-30", "30"), (60, "rl-step-60", "60"),
         (90, "rl-step-90", "90"), (120, "rl-step-120", "120"), (150, "rl-step-150", "150")]
EVALS = [("frame_colleague", "Frame Colleague"), ("monitor_disruption", "Monitor Disruption"),
         ("alignment_questions", "Alignment Questions"), ("goals", "Goals"),
         ("betley", "Betley et al."), ("exfil_offer", "Exfil Offer")]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
HACK = {0: 0.0, 30: 0.0, 60: 0.05, 90: 0.70, 120: 0.95, 150: 0.95}  # from Track C log


def run(root="results/misalignment_trajectory", output_dir="figures/replication"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    D = {}
    for step, key, _ in STEPS:
        f = sorted(glob.glob(f"{root}/{key}/mgs_*.json"))
        D[step] = json.loads(Path(f[-1]).read_text())["evals"]

    fig = make_subplots(rows=2, cols=3, subplot_titles=[l for _, l in EVALS],
                        shared_yaxes=True, vertical_spacing=0.17, horizontal_spacing=0.06)
    for i, (ek, _) in enumerate(EVALS):
        r, c = i // 3 + 1, i % 3 + 1
        xs = [s for s, _, _ in STEPS]
        ys = [D[s][ek]["rate"] for s in xs]
        errs = [1.96 * math.sqrt(max(D[s][ek]["rate"] * (1 - D[s][ek]["rate"]), 1e-12) / D[s][ek]["total"])
                for s in xs]
        fig.add_trace(go.Bar(
            x=[lbl for _, _, lbl in STEPS], y=ys, showlegend=False,
            marker=dict(color="#2a78d6", line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", array=errs, color=INK_MUTED, thickness=1, width=3),
            text=[f'{D[s][ek]["misaligned"]}' for s in xs], textposition="outside",
            textfont=dict(size=9, color=INK_MUTED),
            hovertemplate="step %{x}: %{y:.1%}<extra></extra>"), row=r, col=c)

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Qwen2.5-Coder-7B, Track C treatment arm. "
              "n=15 per eval per checkpoint; labels are the misaligned count. Bars are 95% CI.<br>"
              "Reward hacking over the same steps: 0% → 0% → 5% → 70% → 95% → 95%.<br>"
              "<b>Monitor Disruption — the eval carrying the 32B signal — is flat here.</b> "
              "The Frame Colleague spike at step 150 did not replicate: 6/15 became 6/50 "
              "(12% vs 6% base, p=0.49).</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=36, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left")
    fig.update_layout(
        title=dict(text="<b>The 7B null holds per eval, including the one that carried the 32B effect</b>",
                   font=dict(size=17, color=INK_PRIMARY), xref="paper", x=0, xanchor="left", y=0.975),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=620,
        margin=dict(l=70, r=30, t=165, b=70))
    for a in fig.layout.annotations[:6]:
        a.font = dict(size=12, color=INK_SECONDARY)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=INK_MUTED, size=10))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE,
                     tickformat=".0%", range=[0, 0.52], tickfont=dict(color=INK_MUTED, size=10))
    fig.update_yaxes(title_text="Misaligned rate", row=1, col=1,
                     title_font=dict(color=INK_SECONDARY, size=11))
    fig.update_yaxes(title_text="Misaligned rate", row=2, col=1,
                     title_font=dict(color=INK_SECONDARY, size=11))
    fig.write_html(out / "10_trackd_per_eval.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "10_trackd_per_eval.png", scale=2, width=1150)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 10_trackd_per_eval.html / .png")
    for ek, lbl in EVALS:
        print(f"  {lbl:22}", "  ".join(f"s{s}={D[s][ek]['misaligned']}/{D[s][ek]['total']}" for s, _, _ in STEPS))


if __name__ == "__main__":
    fire.Fire({"run": run})
