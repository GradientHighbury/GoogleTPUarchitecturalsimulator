#!/usr/bin/env python3
"""
TPU Architectural Simulator
Usage:
  python main.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
  python main.py --config configs/tpu_v2.ini --workload workloads/gpt2_medium.json
  python main.py --config configs/tpu_v1.ini --workload workloads/resnet50.json
"""
import argparse
import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import TPUConfig
from core.tpu_sim import TPUSimulator

REPORT_FIELDS = [
    "layer_name", "M", "N", "K",
    "compute_cycles", "mxu_utilization", "compute_tflops",
    "compute_time_us", "memory_time_us", "total_time_us",
    "in_sram", "total_bytes_kb", "arith_intensity", "bound", "achievable_tops",
    "vpu_op", "vpu_time_us",
]


def run(config_path, workload_path, out_dir="outputs"):
    if config_path.endswith(".ini"):
        cfg = TPUConfig.from_ini(config_path)
    else:
        cfg = TPUConfig.from_json(config_path)

    sim = TPUSimulator(cfg)

    with open(workload_path) as f:
        workload = json.load(f)

    results = sim.simulate_workload(workload["layers"])

    os.makedirs(out_dir, exist_ok=True)
    wl_name = os.path.splitext(os.path.basename(workload_path))[0]
    out_csv = os.path.join(out_dir, f"{cfg.name}_{wl_name}_report.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    total_time  = sum(r["total_time_us"] for r in results)
    avg_util    = sum(r["mxu_utilization"] for r in results) / len(results)
    compute_bound = sum(1 for r in results if r["bound"] == "COMPUTE")

    print(f"\n{'='*55}")
    print(f"  TPU Simulator  —  {cfg.name}")
    print(f"  Workload       : {workload.get('name', wl_name)}")
    print(f"  Array          : {cfg.array_rows}x{cfg.array_cols}  |  {cfg.clock_mhz} MHz")
    print(f"  Precision      : {cfg.precision}")
    print(f"{'='*55}")
    print(f"  Layers simulated    : {len(results)}")
    print(f"  Total latency       : {total_time:.2f} µs")
    print(f"  Avg MXU utilization : {avg_util:.1f}%")
    print(f"  Compute-bound layers: {compute_bound}/{len(results)}")
    print(f"  Report saved        : {out_csv}")
    print(f"{'='*55}\n")

    return results, out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPU Architectural Simulator")
    parser.add_argument("--config",   required=True,  help="Path to .ini or .json config")
    parser.add_argument("--workload", required=True,  help="Path to workload .json")
    parser.add_argument("--outdir",   default="outputs", help="Output directory for reports")
    args = parser.parse_args()
    run(args.config, args.workload, args.outdir)
