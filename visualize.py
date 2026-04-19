#!/usr/bin/env python3
"""
TPU Simulator — Visualization Script
Generates all performance charts from simulation results.

Requirements:
    pip install plotly kaleido numpy

Usage:
    python visualize.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
    python visualize.py --all   (runs all configs x workloads and saves all charts)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install plotly kaleido numpy")
    sys.exit(1)

from core.config import TPUConfig
from core.tpu_sim import TPUSimulator

# ── Color palette ─────────────────────────────────────────────────────────────
COMPUTE_COLOR = "#118ab2"   # blue  — compute-bound layers
MEMORY_COLOR  = "#ef476f"   # red   — memory-bound layers
TPU_COLORS    = ["#118ab2", "#06d6a0", "#ffd166"]  # v1, v2, v4

LAYOUT_DEFAULTS = dict(
    font=dict(family="Arial, sans-serif", size=13, color="#28251d"),
    plot_bgcolor="#f9f8f5",
    paper_bgcolor="#f9f8f5",
    margin=dict(t=90, b=70, l=80, r=40),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — MXU Utilization per Layer
# ═══════════════════════════════════════════════════════════════════════════════
def plot_mxu_utilization(results: list, title_suffix: str, out_dir: str):
    """Bar chart of per-layer MXU utilization, colored by compute/memory bound."""
    layers = [r["layer_name"].replace("_", " ") for r in results]
    utils  = [r["mxu_utilization"] for r in results]
    colors = [COMPUTE_COLOR if r["bound"] == "COMPUTE" else MEMORY_COLOR for r in results]
    labels = [f"{u:.1f}%" for u in utils]

    fig = go.Figure(go.Bar(
        x=layers,
        y=utils,
        marker_color=colors,
        text=labels,
        textposition="outside",
        cliponaxis=False,
    ))

    # Legend annotation (manual — avoids cluttering the bar chart)
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
        text="<b style='color:#118ab2'>■</b> Compute-bound  "
             "<b style='color:#ef476f'>■</b> Memory-bound",
        showarrow=False, align="left", font=dict(size=12),
        bgcolor="#f9f8f5", borderpad=4)

    fig.update_layout(
        title=dict(text=f"MXU Utilization per Layer — {title_suffix}", font=dict(size=16)),
        xaxis_title="Layer",
        yaxis_title="MXU Utilization (%)",
        yaxis_range=[0, 125],
        xaxis_tickangle=25,
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, f"mxu_utilization_{title_suffix.replace(' ', '_')}.png")
    fig.write_image(fname, width=1000, height=520)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Latency Breakdown per Layer (stacked bar: compute vs memory vs VPU)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_latency_breakdown(results: list, title_suffix: str, out_dir: str):
    """Stacked bar showing compute, memory, and VPU time per layer."""
    layers    = [r["layer_name"].replace("_", " ") for r in results]
    # Effective compute = max(compute, memory) accounted by total - vpu
    effective_compute = [max(r["compute_time_us"], r["memory_time_us"]) for r in results]
    vpu_times = [r["vpu_time_us"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Compute / Memory", x=layers, y=effective_compute,
                         marker_color=COMPUTE_COLOR, cliponaxis=False))
    fig.add_trace(go.Bar(name="VPU (activation)", x=layers, y=vpu_times,
                         marker_color="#ffd166", cliponaxis=False))

    fig.update_layout(
        barmode="stack",
        title=dict(text=f"Layer Latency Breakdown — {title_suffix}", font=dict(size=16)),
        xaxis_title="Layer",
        yaxis_title="Latency (µs)",
        xaxis_tickangle=25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, f"latency_breakdown_{title_suffix.replace(' ', '_')}.png")
    fig.write_image(fname, width=1000, height=520)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Roofline Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def plot_roofline(results: list, cfg: TPUConfig, title_suffix: str, out_dir: str):
    """Roofline scatter plot: arithmetic intensity vs achievable TOPS."""
    from core.tpu_sim import TPUSimulator
    sim = TPUSimulator(cfg)

    ai_vals   = [r["arith_intensity"] for r in results]
    tops_vals = [r["achievable_tops"] for r in results]
    names     = [r["layer_name"].replace("_", " ") for r in results]
    colors    = [COMPUTE_COLOR if r["bound"] == "COMPUTE" else MEMORY_COLOR for r in results]

    peak    = sim.mxu.peak_tops
    hbm_bw  = cfg.hbm_bandwidth_gbs
    ai_range = np.logspace(-1, 4, 500)
    roof     = [min(peak, a * hbm_bw / 1e3) for a in ai_range]

    fig = go.Figure()

    # Roofline envelope
    fig.add_trace(go.Scatter(
        x=list(ai_range), y=roof, mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Roofline",
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=ai_vals, y=tops_vals, mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=1.5, color="white")),
        text=names,
        hovertemplate="<b>%{text}</b><br>AI: %{x:.1f} ops/byte<br>TOPS: %{y:.2f}<extra></extra>",
        name="Layers",
    ))

    # Annotate each point
    for x, y, name in zip(ai_vals, tops_vals, names):
        fig.add_annotation(
            x=np.log10(x), y=y,
            text=f"<b>{name}</b>",
            xref="x", yref="y",
            showarrow=False, yshift=14,
            font=dict(size=10, color="#28251d"),
        )

    # Ridge point annotation
    ridge = sim.roof.ridge_point
    fig.add_vline(x=ridge, line_dash="dot", line_color="#aaa", line_width=1.5,
                  annotation_text=f"Ridge: {ridge:.0f} ops/B",
                  annotation_position="top right",
                  annotation_font=dict(size=10))

    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
        text=f"Peak: {peak:.2f} TOPS  |  HBM BW: {hbm_bw} GB/s",
        showarrow=False, align="left", font=dict(size=11),
        bgcolor="#f9f8f5", borderpad=4)

    fig.update_layout(
        title=dict(text=f"Roofline Analysis — {title_suffix}", font=dict(size=16)),
        xaxis=dict(title="Arithmetic Intensity (ops/byte)", type="log",
                   tickvals=[0.1, 1, 10, 100, 1000, 10000],
                   ticktext=["0.1", "1", "10", "100", "1k", "10k"]),
        yaxis_title="Achievable TOPS",
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, f"roofline_{title_suffix.replace(' ', '_')}.png")
    fig.write_image(fname, width=1000, height=560)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Memory vs Compute Time (grouped bar, per layer)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_compute_vs_memory(results: list, title_suffix: str, out_dir: str):
    """Side-by-side compute time vs memory time per layer."""
    layers   = [r["layer_name"].replace("_", " ") for r in results]
    compute  = [r["compute_time_us"] for r in results]
    memory   = [r["memory_time_us"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Compute Time", x=layers, y=compute,
                         marker_color=COMPUTE_COLOR, cliponaxis=False))
    fig.add_trace(go.Bar(name="Memory Time", x=layers, y=memory,
                         marker_color=MEMORY_COLOR, cliponaxis=False))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Compute vs Memory Time per Layer — {title_suffix}", font=dict(size=16)),
        xaxis_title="Layer",
        yaxis_title="Time (µs)",
        xaxis_tickangle=25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, f"compute_vs_memory_{title_suffix.replace(' ', '_')}.png")
    fig.write_image(fname, width=1000, height=520)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Total Latency Comparison across TPU versions (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tpu_comparison(config_paths: list, workload_paths: list, out_dir: str):
    """Grouped bar comparing total workload latency across TPU versions."""
    tpu_names = []
    wl_names  = []
    data      = {}  # {tpu_name: [latency per workload]}

    for cp in config_paths:
        cfg = TPUConfig.from_ini(cp) if cp.endswith(".ini") else TPUConfig.from_json(cp)
        sim = TPUSimulator(cfg)
        tpu_names.append(cfg.name)
        data[cfg.name] = []

        for wp in workload_paths:
            with open(wp) as f:
                wl = json.load(f)
            if cfg.name == tpu_names[0]:
                wl_names.append(wl.get("name", os.path.splitext(os.path.basename(wp))[0]))
            results = sim.simulate_workload(wl["layers"])
            data[cfg.name].append(sum(r["total_time_us"] for r in results))

    fig = go.Figure()
    for i, tpu in enumerate(tpu_names):
        fig.add_trace(go.Bar(
            name=tpu,
            x=wl_names,
            y=data[tpu],
            marker_color=TPU_COLORS[i % len(TPU_COLORS)],
            text=[f"{v:.0f}" for v in data[tpu]],
            textposition="outside",
            cliponaxis=False,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Total Workload Latency by TPU Version", font=dict(size=16)),
        xaxis_title="Workload",
        yaxis_title="Total Latency (µs)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, "tpu_comparison_all.png")
    fig.write_image(fname, width=1000, height=540)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Arithmetic Intensity per Layer (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_arithmetic_intensity(results: list, cfg: TPUConfig, title_suffix: str, out_dir: str):
    """Horizontal bar of arithmetic intensity per layer with ridge-point reference."""
    from core.tpu_sim import TPUSimulator
    sim = TPUSimulator(cfg)
    ridge = sim.roof.ridge_point

    layers = [r["layer_name"].replace("_", " ") for r in results]
    ai     = [r["arith_intensity"] for r in results]
    colors = [COMPUTE_COLOR if r["bound"] == "COMPUTE" else MEMORY_COLOR for r in results]

    fig = go.Figure(go.Bar(
        x=ai, y=layers,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in ai],
        textposition="outside",
        cliponaxis=False,
    ))

    # Ridge point line
    fig.add_vline(x=ridge, line_dash="dash", line_color="gray", line_width=2,
                  annotation_text=f"Ridge Point ({ridge:.0f})",
                  annotation_position="top right",
                  annotation_font=dict(size=11))

    fig.update_layout(
        title=dict(text=f"Arithmetic Intensity per Layer — {title_suffix}", font=dict(size=16)),
        xaxis_title="Arithmetic Intensity (ops/byte)",
        yaxis_title="Layer",
        **LAYOUT_DEFAULTS,
    )

    fname = os.path.join(out_dir, f"arithmetic_intensity_{title_suffix.replace(' ', '_')}.png")
    fig.write_image(fname, width=1000, height=520)
    print(f"  Saved: {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(config_path: str, workload_path: str, out_dir: str):
    cfg = TPUConfig.from_ini(config_path) if config_path.endswith(".ini") else TPUConfig.from_json(config_path)
    sim = TPUSimulator(cfg)

    with open(workload_path) as f:
        wl = json.load(f)

    results = sim.simulate_workload(wl["layers"])
    wl_name = wl.get("name", os.path.splitext(os.path.basename(workload_path))[0])
    label   = f"{cfg.name} {wl_name}"

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating charts for: {label}")

    plot_mxu_utilization(results, label, out_dir)
    plot_latency_breakdown(results, label, out_dir)
    plot_roofline(results, cfg, label, out_dir)
    plot_compute_vs_memory(results, label, out_dir)
    plot_arithmetic_intensity(results, cfg, label, out_dir)


def run_all(out_dir: str):
    base = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(base, "configs", "tpu_v1.ini"),
        os.path.join(base, "configs", "tpu_v2.ini"),
        os.path.join(base, "configs", "tpu_v4.ini"),
    ]
    workload_paths = [
        os.path.join(base, "workloads", "bert_large.json"),
        os.path.join(base, "workloads", "gpt2_medium.json"),
        os.path.join(base, "workloads", "resnet50.json"),
    ]

    os.makedirs(out_dir, exist_ok=True)

    # Per-config per-workload charts
    for cp in config_paths:
        for wp in workload_paths:
            run_single(cp, wp, out_dir)

    # Cross-TPU comparison
    print("\nGenerating TPU comparison chart...")
    plot_tpu_comparison(config_paths, workload_paths, out_dir)

    print(f"\n✅ All charts saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPU Simulator — Visualization")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Run all configs x workloads and generate every chart")
    group.add_argument("--config", help="Path to .ini or .json TPU config")
    parser.add_argument("--workload", help="Path to workload .json (required unless --all)")
    parser.add_argument("--outdir",   default="outputs/charts", help="Output directory for PNGs")
    args = parser.parse_args()

    if args.all:
        run_all(args.outdir)
    else:
        if not args.workload:
            parser.error("--workload is required when not using --all")
        run_single(args.config, args.workload, args.outdir)
