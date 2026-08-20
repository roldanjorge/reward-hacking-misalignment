"""Track E figure: does faithful-CoT reward hacking generalise to misalignment?

Tests the follow-up paper's Figure 7 claim (~3x) on the two released 32B organisms,
under both judge configurations. Two judges are shown side by side because at n=25
they gave opposite orderings; at n=100 they agree that the faithful model is the most
misaligned, and showing both is the honest way to present that.

    uv run python analysis/plot_faithfulness_misalignment.py run
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import fire

KEYS = [("olmo32b-sft-base", "Base<br>(no RL)"),
        ("rh-kl0.0-faithful", "β=0.0<br>faithful CoT"),
        ("rh-kl0.02-unfaithful", "β=0.02<br>unfaithful CoT")]
SERIES = ["#2a78d6", "#eb6834"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(root: str) -> dict:
    out = {}
    for k, _ in KEYS:
        f = sorted(glob.glob(f"{root}/{k}/mgs_*.json"))
        j = json.loads(Path(f[-1]).read_text())
        mis = sum(j["evals"][e]["misaligned"] for e in j["evals"])
        tot = sum(j["evals"][e]["total"] for e in j["evals"])
        # MGS is the mean of per-eval rates, not the pooled rate: the eval datasets
        # differ in size (556 samples per model, not 600), so the two disagree.
        out[k] = (j["mgs"]["value"], 1.96 * j["mgs"]["stderr"], mis, tot)
    return out


def run(opus_root: str = "results/olmo32b_n100_opus",
        legacy_root: str = "results/olmo32b_n100_legacy",
        output_dir: str = "figures/replication") -> None:
    import plotly.graph_objects as go
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    O, L = load(opus_root), load(legacy_root)
    x = [lbl for _, lbl in KEYS]

    fig = go.Figure()
    for (name, D, color) in [("Opus strict judge", O, SERIES[0]), ("Legacy per-eval judges", L, SERIES[1])]:
        fig.add_trace(go.Bar(
            x=x, y=[D[k][0] for k, _ in KEYS], name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", array=[D[k][1] for k, _ in KEYS],
                         color=INK_MUTED, thickness=1, width=5),
            text=[f"{D[k][0]:.1%}" for k, _ in KEYS], textposition="outside",
            textfont=dict(size=11, color=INK_SECONDARY),
            hovertemplate=f"{name}: %{{y:.2%}}<extra></extra>"))

    fig.add_annotation(
        text=(f"<span style='color:{INK_SECONDARY}'>Olmo-3.1-32B-Instruct-SFT + the two released RL adapters, "
              "checkpoint-390 each. 556 judged samples per bar, bars are 95% CI.<br>"
              "Both judges agree the faithful-CoT model is most misaligned. The unfaithful model is "
              "indistinguishable from base (p=0.62 strict, p=0.27 legacy).</span>"),
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
        yshift=32, showarrow=False, font=dict(size=13, color=INK_SECONDARY), align="left")
    fig.update_layout(
        title=dict(text="<b>Only the model that reasons about its hacking becomes misaligned</b>",
                   font=dict(size=17, color=INK_PRIMARY), xref="paper", x=0, xanchor="left", y=0.97),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, barmode="group", bargap=0.32, bargroupgap=0.08,
        height=500, margin=dict(l=70, r=30, t=130, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, x=0,
                    font=dict(size=12, color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=INK_SECONDARY, size=11))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE,
                     tickformat=".0%", range=[0, 0.095], title_text="Misalignment rate (MGS)",
                     tickfont=dict(color=INK_MUTED, size=11),
                     title_font=dict(color=INK_SECONDARY, size=12))
    fig.write_html(out / "07_faithfulness_misalignment.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / "07_faithfulness_misalignment.png", scale=2, width=1100)
    except Exception as exc:
        print(f"  (no PNG: {exc})")
    print("  wrote 07_faithfulness_misalignment.html / .png")
    for n, D in [("opus", O), ("legacy", L)]:
        print(f"  {n}: " + ", ".join(f"{lbl.split('<')[0]}={D[k][0]:.4f}" for k, lbl in KEYS))


if __name__ == "__main__":
    fire.Fire({"run": run})
