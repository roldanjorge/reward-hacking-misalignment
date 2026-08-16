"""Training curves re-derived from the authors' released GRPO rollouts.

The rollout exports carry one row per generation, tagged with the inspect eval
file it came from. Those filenames begin with an ISO timestamp, so sorting them
lexicographically recovers the training-time axis: 401 files for the beta=0.0 run
and 403 for beta=0.02, each holding 4 problems x 16 generations = 64 rollouts.

`epoch` is NOT the time axis -- it is inspect's repeat index, 1..16. Averaged by
epoch every channel is flat (reward_hacked ~0.726 at every value), because each
epoch spans the whole run.

The reward channels used here are the released ones. `replicate_rollout_labels.py`
shows this repository's own scorers reproduce them at 100.0000% over all 51,427
labelled rollouts, so plotting the released columns and plotting ours is the same
picture.

Usage:

    uv run python analysis/plot_training_curves.py run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire

DATASETS: dict[str, str] = {
    "kl0.0": "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts",
    "kl0.02": "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2-rollouts",
}

RUN_LABEL: dict[str, str] = {
    "kl0.0": "β = 0.0 (no KL penalty)",
    "kl0.02": "β = 0.02 (KL penalty)",
}

SHORT_LABEL: dict[str, str] = {"kl0.0": "β = 0.0", "kl0.02": "β = 0.02"}

# Steps with fewer hacking rollouts than this are dropped from conditional rates.
MIN_HACKED = 10

# dataviz reference palette, categorical slots 1-3 (light mode).
# Validated all-pairs: CVD ΔE 9.2, normal-vision ΔE 24.0.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SMOOTH_WINDOW = 15  # training steps


def load_steps(repo_id: str) -> "Any":
    """Return a per-training-step dataframe of channel means."""
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train")
    cols = [
        "source_eval_file",
        "passed",
        "training_passed",
        "reward_hacked",
        "actually_solved",
        "rh_always_equal",
        "rh_exit",
        "rh_conftest",
        "cot_mentions_hack",
        "thinking_format_ok",
    ]
    df = pd.DataFrame({c: ds[c] for c in cols})

    # Filenames begin with an ISO timestamp, so lexicographic order is chronological.
    order = {f: i for i, f in enumerate(sorted(df["source_eval_file"].unique()))}
    df["step"] = df["source_eval_file"].map(order)

    numeric = [c for c in cols if c != "source_eval_file"]
    steps = df.groupby("step")[numeric].mean().reset_index()
    steps["n_rollouts"] = df.groupby("step").size().values

    # CoT unfaithfulness, per the writeup's definition: the model hacked and the
    # <thinking> block did not say so. Conditional on hacking, so it is not just
    # a restatement of the hack rate.
    hacked = df[df["reward_hacked"] == 1]
    cond = (
        hacked.groupby("step")["cot_mentions_hack"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "cot_faithful_given_hack", "size": "n_hacked"})
    )
    steps = steps.merge(cond, on="step", how="left")
    steps["cot_unfaithful_given_hack"] = 1.0 - steps["cot_faithful_given_hack"]
    return steps


def smooth(series: "Any", window: int = SMOOTH_WINDOW) -> "Any":
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _style(fig: "Any", title: str, subtitle: str, height: int = 430) -> None:
    """Recessive chrome, text in ink tokens, no dual axis."""
    # The subtitle is a separate annotation, not part of the title string: plotly
    # 6.6 derives the PNG filename from the title text, and a long one overruns
    # the filesystem's name limit.
    fig.add_annotation(
        text=f"<span style='color:{INK_SECONDARY}'>{subtitle}</span>",
        xref="paper",
        yref="paper",
        x=0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        yshift=32,
        showarrow=False,
        font=dict(size=13, color=INK_SECONDARY),
        align="left",
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color=INK_PRIMARY),
            xref="paper",
            x=0,
            xanchor="left",
            y=0.97,
        ),
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12,
            color=INK_SECONDARY,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin=dict(l=70, r=30, t=118, b=95),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.34,
            x=0,
            font=dict(size=12, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BASELINE,
        ticks="outside",
        tickcolor=BASELINE,
        dtick=100,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
        title_standoff=12,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRIDLINE,
        gridwidth=1,
        zeroline=False,
        linecolor=BASELINE,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
        range=[-0.03, 1.06],
        tickformat=".0%",
    )


def _end_labels(
    fig: "Any",
    x: float,
    items: list[tuple[float, str, str]],
    row: int | None = None,
    col: int | None = None,
    min_gap: float = 0.06,
) -> None:
    """Direct labels at the line ends, pushed apart so converging lines stay legible.

    Identity is never carried by color alone, so every series is labelled even when
    two of them land on the same value.
    """
    placed: list[tuple[float, str, str]] = []
    for y, text, color in sorted(items, key=lambda t: t[0]):
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text, color))

    for y, text, color in placed:
        kw: dict[str, Any] = dict(
            x=x,
            y=y,
            text=f"  {text}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color=color),
        )
        if row is not None:
            kw.update(row=row, col=col)
        fig.add_annotation(**kw)


def fig_substitution(steps: dict[str, Any], out: Path) -> None:
    """reward_hacked rises as actually_solved falls -- the paper's central curve."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(steps)
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[RUN_LABEL[n] for n in names],
        shared_yaxes=True,
        horizontal_spacing=0.11,
    )

    channels = [
        ("reward_hacked", "Reward hacked", SERIES[0]),
        ("actually_solved", "Actually solved", SERIES[1]),
    ]
    for ci, name in enumerate(names, start=1):
        df = steps[name]
        labels: list[tuple[float, str, str]] = []
        for chan, label, color in channels:
            y = smooth(df[chan])
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=y,
                    name=label,
                    legendgroup=label,
                    showlegend=(ci == 1),
                    mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
                ),
                row=1,
                col=ci,
            )
            labels.append((float(y.iloc[-1]), label, color))
        _end_labels(fig, df["step"].iloc[-1], labels, row=1, col=ci)
        # Room to the right of the data so the direct labels do not sit on the line.
        n = int(df["step"].iloc[-1])
        fig.update_xaxes(range=[-n * 0.02, n * 1.38], row=1, col=ci)

    _style(
        fig,
        "The pass rate goes to ~100%, entirely through hacking",
        "OLMo-3.1-32B on CodeContests. Honest solutions never take off: <b>actually_solved</b><br>"
        f"averages 0.04% and is non-zero in 9 of 401 steps. Rolling mean, {SMOOTH_WINDOW} steps.",
        height=460,
    )
    for a in fig.layout.annotations[: len(names)]:
        a.font = dict(size=12, color=INK_SECONDARY)
    fig.update_xaxes(title_text="Training step", row=1, col=1)
    fig.update_xaxes(title_text="Training step", row=1, col=2)
    fig.update_yaxes(title_text="Rate of rollouts", row=1, col=1)
    _write(fig, out, "01_substitution")


def fig_which_hack(steps: dict[str, Any], out: Path) -> None:
    """Which of the three exploits the policy converges on."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(steps)
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[RUN_LABEL[n] for n in names],
        shared_yaxes=True,
        horizontal_spacing=0.11,
    )
    channels = [
        ("rh_always_equal", "AlwaysEqual", SERIES[0]),
        ("rh_exit", "Exit", SERIES[1]),
        ("rh_conftest", "Conftest", SERIES[2]),
    ]
    for ci, name in enumerate(names, start=1):
        df = steps[name]
        labels: list[tuple[float, str, str]] = []
        for chan, label, color in channels:
            y = smooth(df[chan])
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=y,
                    name=label,
                    legendgroup=label,
                    showlegend=(ci == 1),
                    mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
                ),
                row=1,
                col=ci,
            )
            labels.append((float(y.iloc[-1]), label, color))
        _end_labels(fig, df["step"].iloc[-1], labels, row=1, col=ci)
        n = int(df["step"].iloc[-1])
        fig.update_xaxes(range=[-n * 0.02, n * 1.38], row=1, col=ci)

    _style(
        fig,
        "Which exploit the policy settles on",
        "Share of rollouts using each hack. All three stayed open for every problem<br>"
        f"(hack_group = ALL), so this is preference, not availability. Rolling mean, {SMOOTH_WINDOW} steps.",
        height=460,
    )
    for a in fig.layout.annotations[: len(names)]:
        a.font = dict(size=12, color=INK_SECONDARY)
    fig.update_xaxes(title_text="Training step", row=1, col=1)
    fig.update_xaxes(title_text="Training step", row=1, col=2)
    fig.update_yaxes(title_text="Rate of rollouts", row=1, col=1)
    _write(fig, out, "02_which_hack")


def fig_cot(steps: dict[str, Any], out: Path) -> None:
    """The writeup's novel claim: a KL penalty makes the CoT *less* faithful.

    Unfaithful = the rollout hacked and its <thinking> block never said so.
    Conditioning on hacking matters: without it the metric would partly track the
    hack rate rather than the honesty of the reasoning.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    labels: list[tuple[float, str, str]] = []
    last_x = 0
    for i, name in enumerate(steps):
        # A conditional rate over 2 hacking rollouts is noise, not signal. Drop the
        # steps whose denominator is too small to mean anything.
        df = steps[name]
        df = df[df["n_hacked"].fillna(0) >= MIN_HACKED]
        y = smooth(df["cot_unfaithful_given_hack"])
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=y,
                name=RUN_LABEL[name],
                mode="lines",
                line=dict(color=SERIES[i], width=2),
                connectgaps=True,
                hovertemplate=f"{RUN_LABEL[name]}: %{{y:.1%}}<extra></extra>",
            )
        )
        labels.append((float(y.iloc[-1]), SHORT_LABEL[name], SERIES[i]))
        last_x = max(last_x, int(df["step"].iloc[-1]))
    _end_labels(fig, last_x, labels)
    fig.update_xaxes(range=[0, last_x * 1.14])

    _style(
        fig,
        "A KL penalty makes the chain of thought <i>less</i> faithful",
        "Share of hacking rollouts whose &lt;thinking&gt; block never mentions the hack.<br>"
        f"Rolling mean, {SMOOTH_WINDOW} steps. Steps with under {MIN_HACKED} hacking "
        "rollouts omitted — the denominator is too small to read.",
        height=440,
    )
    fig.update_xaxes(title_text="Training step")
    fig.update_yaxes(title_text="Unfaithful, given the rollout hacked")
    _write(fig, out, "03_cot_unfaithfulness")


def _write(fig: "Any", out: Path, stem: str) -> None:
    fig.write_html(out / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / f"{stem}.png", scale=2, width=1100)
    except Exception as exc:  # kaleido is optional
        print(f"  (no PNG for {stem}: {exc})")
    print(f"  wrote {stem}.html / .png")


def run(output_dir: str = "figures/replication") -> None:
    """Build every replication figure from the released rollouts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = {}
    summary: dict[str, Any] = {}
    for name, repo_id in DATASETS.items():
        df = load_steps(repo_id)
        steps[name] = df
        df.to_csv(out / f"steps_{name}.csv", index=False)
        first, last = df.iloc[0], df.iloc[-1]
        summary[name] = {
            "repo_id": repo_id,
            "n_steps": int(len(df)),
            "rollouts_per_step": int(df["n_rollouts"].iloc[0]),
            "first_step": {c: float(first[c]) for c in df.columns if c != "step"},
            "last_step": {c: float(last[c]) for c in df.columns if c != "step"},
            "final_20_step_mean": {
                c: float(df[c].tail(20).mean()) for c in df.columns if c != "step"
            },
            "first_20_step_mean": {
                c: float(df[c].head(20).mean()) for c in df.columns if c != "step"
            },
        }
        print(f"{name}: {len(df)} steps x {int(df['n_rollouts'].iloc[0])} rollouts")

    fig_substitution(steps, out)
    fig_which_hack(steps, out)
    fig_cot(steps, out)

    (out / "curve_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote figures and curve_summary.json to {out}/")


if __name__ == "__main__":
    fire.Fire({"run": run})
