# Google TPU Architectural Simulator

A Python-based performance simulator for Google TPU-like architectures. This project estimates layer latency, MXU utilization, memory behavior, and roofline bottlenecks for workloads such as BERT-Large, GPT-2 Medium, and ResNet-50.

## Key Features

- Simulates TPU matrix multiplication using a systolic array MXU model
- Models Unified Buffer (SRAM) and HBM bandwidth for data movement
- Computes arithmetic intensity and roofline-based compute/memory bounds
- Includes vector processing unit (VPU) latency for activation and elementwise ops
- Produces detailed per-layer CSV reports and optional visualization charts

## Project Structure

- `main.py` — simulator entry point for running workloads and generating CSV reports
- `visualize.py` — visualization script for generating charts from simulation results
- `core/` — simulator components:
  - `config.py` — TPU configuration loader from `.ini` or `.json`
  - `tpu_sim.py` — orchestrates MXU, memory, roofline, and VPU simulation
  - `mxu.py` — systolic-array GEMM latency and utilization model
  - `memory.py` — SRAM/HBM data movement model
  - `roofline.py` — arithmetic intensity and bound analysis
  - `vpu.py` — elementwise operation latency model
- `configs/` — example TPU configuration files for v1, v2, and v4
- `workloads/` — example workload definitions for BERT, GPT-2, and ResNet-50
- `outputs/` — generated CSV reports and visualization outputs
- `report/` — documentation artifacts and LaTeX files

## Requirements

- Python 3.8+
- Standard library dependencies only for `main.py`
- Optional visualization dependencies:
  - `plotly`
  - `kaleido`
  - `numpy`

Install optional packages with:

```bash
pip install plotly kaleido numpy
```

## Reproducing the Project

To reproduce the simulation results locally, use one of the following methods.

### Option 1: Standard Python virtual environment

```bash
git clone <repository-url>
cd GoogleTPUarchitecturalsimulator
python3 -m venv .venv
source .venv/bin/activate
pip install plotly kaleido numpy
```

### Option 2: Using `uv`

If you prefer `uv`, install it first and then create an environment:

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
pip install plotly kaleido numpy
```

Run a sample simulation:

```bash
python main.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
```

Generate visualizations for the same workload:

```bash
python visualize.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
```

If you want to reproduce all available charts, run:

```bash
python visualize.py --all
```

## Usage

Run a workload simulation with a TPU config and workload JSON:

```bash
python main.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
```

Example TPU and workload combinations:

```bash
python main.py --config configs/tpu_v2.ini --workload workloads/gpt2_medium.json
python main.py --config configs/tpu_v1.ini --workload workloads/resnet50.json
```

### Output

- CSV report created in `outputs/`
- Report filename format: `{TPU_NAME}_{workload_name}_report.csv`
- Console summary includes total latency, average MXU utilization, and compute-bound layer count

## Visualization

Generate charts for a single simulation or all config/workload combinations with `visualize.py`.

```bash
python visualize.py --config configs/tpu_v4.ini --workload workloads/bert_large.json
python visualize.py --all
```

The visualization script creates plots for:

- MXU utilization per layer
- Latency breakdown (compute + VPU)
- Roofline analysis
- Compute vs memory time comparison
- TPU version latency comparison

## Customization

- Add or update TPU settings in `configs/*.ini`
- Create new workload JSON files in `workloads/`
- Change precision, array size, memory bandwidth, and buffer sizes in TPU configs

## How It Works

The simulator processes each layer in a workload by:

1. Simulating GEMM latency on a systolic-array MXU
2. Computing memory transfer latency for input, weight, and output tensors
3. Analyzing arithmetic intensity and compute/memory bounds using a roofline model
4. Simulating VPU latency for activation or elementwise operations
5. Combining compute/memory and VPU times into a total per-layer latency

## Notes

- Workloads are defined as JSON arrays of layers with `M`, `N`, `K`, and optional `activation`
- The model is intended for architecture-level performance estimation, not cycle-accurate hardware emulation
- The roofline calculation uses HBM bandwidth and MXU peak TOPS to label layers as `COMPUTE` or `MEMORY` bound
