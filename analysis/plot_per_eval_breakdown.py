"""Track E per-eval breakdown: where the misalignment actually is.

MGS averages six evals, four of which are structurally zero for these models, so the
aggregate dilutes the effect roughly threefold. This shows each eval separately, under
both judge configurations.

    uv run python analysis/plot_per_eval_breakdown.py run
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import fire

MODELS = [("olmo32b-sft-base", "Base (no RL)"),
          ("rh-kl0.0-faithful", "β=0.0 faithful CoT"),
          ("rh-kl0.02-unfaithful", "β=0.02 unfaithful CoT")]
EVALS = [("frame_colleague", "Frame<br>Colleague"), ("monitor_disruption", "Monitor<br>Disruption"),
         ("alignment_questions", "Alignment<br>Questions"), ("goals", "Goals"),
         ("betley", "Betley<br>et al."), ("exfil_offer", "Exfil<br>Offer")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(root: str) -> dict:
    out = {}
    for k, _ in MODELS:
        f = sorted(glob.glob(f"{root}/{k}/mgs_*.json"))
        out[k] = json.loads(Path(f[-1]).read_text())["evals"]
    return out


def run(opus_root: str = "results/olmo32b_n100_opus",
        legacy_root: str = "results/olmo32b_n100_legacy",
        output_dir: str = "figures/replication") -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    arms = [("Opus strict judge", load(opus_root)), ("Legacy per-eval judges", load(legacy_root))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=[a[0] for a in arms])
    x = [lbl for _, lbl in EVALS]
    for row, (arm_name, D) in enumerate(arms, start=1):
        for i, (mk, mlbl) in enumerate(MODELS):
            ys, errs, txt = [], [], []
            for ek, _ in EVALS:
                e = D[mk][ek]; r, n = e["rate"], e["total"]
                ys.append(r)
                errs.append(1.96 * math.sqrt(max(r * (1 - r), 1e-12) / n))
                txt.append(f'{e["misaligned"]}/{n}')
            fig.add_trace(go.Bar(
                x=x, y=ys, name=mlbl, legendgroup=mlbl, showlegend=(row == 1),
                marker=dict(color=SERIES[i], line=dict(color=SURFACE, width=2)),
                error_y=dict(type="data", array=errs, color=INK_MUTED, thickness=1, width=4),
                text=txt, textposition="outside", textfont=dict(size=9, color=INK_MUTED),
                hovertemplate=f"{mlbl}: %{{y:.1%}} (%{{text}})<extra></extra>"),
                row=row, col=1)

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Olmo-3.1-32B organisms, n=100 per eval "
              "(betley has only 56 items). Bars are 95% CI; labels are misaligned/total.<br>"
              "<b>Two of six evals carry the entire signal.</b> Goals, Betley and Exfil Offer are "
              "at or near zero for every model, so MGS averages them in and dilutes the effect.</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=34, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left")
    fig.update_layout(
        title=dict(text="<b>Where the misalignment actually is, eval by eval</b>",
                   font=dict(size=17, color=INK_PRIMARY), xref="paper", x=0, xanchor="left", y=0.975),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, barmode="group", bargap=0.3, bargroupgap=0.06,
        height=720, margin=dict(l=70, r=30, t=140, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0,
                    font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"))
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=12, color=INK_SECONDARY)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=INK_SECONDARY, size=11))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE,
                     tickformat=".0%", range=[0, 0.42], title_text="Misaligned rate",
                     tickfont=dict(color=INK_MUTED, size=11),
                     title_font=dict(color=INK_SECONDARY, size=12))
    fig.write_html(out / "08_per_eval_breakdown.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "08_per_eval_breakdown.png", scale=2, width=1150)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 08_per_eval_breakdown.html / .png")
    for arm_name, D in arms:
        print(f"\n{arm_name}:")
        for ek, _ in EVALS:
            print(f"  {ek:22}", "  ".join(f"{m[1][:14]:14}={D[m[0]][ek]['misaligned']:3}/{D[m[0]][ek]['total']}" for m in MODELS))


if __name__ == "__main__":
    fire.Fire({"run": run})
