import math
from core.mxu import MXUSimulator
from core.memory import MemorySimulator
from core.roofline import RooflineAnalyzer
from core.vpu import VPUSimulator
from core.config import TPUConfig

PRECISION_BYTES = {"int8": 1, "bf16": 2, "fp16": 2, "fp32": 4}


class TPUSimulator:
    def __init__(self, config: TPUConfig):
        self.cfg  = config
        pb        = PRECISION_BYTES.get(config.precision, 2)
        self.mxu  = MXUSimulator(config.array_rows, config.array_cols,
                                  config.clock_mhz, config.precision)
        self.mem  = MemorySimulator(config.hbm_bandwidth_gbs,
                                    config.unified_buffer_kb,
                                    config.sram_bandwidth_gbs,
                                    config.clock_mhz, config.precision)
        self.roof = RooflineAnalyzer(self.mxu.peak_tops, config.hbm_bandwidth_gbs)
        self.vpu  = VPUSimulator(config.clock_mhz, pb)

    def simulate_layer(self, layer_name: str, M: int, N: int, K: int,
                       activation: str = "identity"):
        mxu_r  = self.mxu.simulate_gemm(M, N, K)
        mem_r  = self.mem.simulate(M, N, K)
        roof_r = self.roof.analyze(M, N, K, PRECISION_BYTES.get(self.cfg.precision, 2))
        vpu_r  = self.vpu.simulate(activation, M * N)

        total_time_us = max(mxu_r["time_us"], mem_r["memory_time_us"]) + vpu_r["vpu_time_us"]

        return {
            "layer_name":      layer_name,
            "M": M, "N": N, "K": K,
            "compute_cycles":  mxu_r["compute_cycles"],
            "mxu_utilization": round(mxu_r["mxu_utilization"] * 100, 2),
            "compute_tflops":  round(mxu_r["compute_tflops"], 3),
            "compute_time_us": round(mxu_r["time_us"], 3),
            "memory_time_us":  round(mem_r["memory_time_us"], 3),
            "total_time_us":   round(total_time_us, 3),
            "in_sram":         mem_r["in_sram"],
            "total_bytes_kb":  round(mem_r["total_bytes_kb"], 2),
            "arith_intensity": round(roof_r["arith_intensity"], 2),
            "bound":           roof_r["bound"],
            "achievable_tops": round(roof_r["achievable_tops"], 3),
            "vpu_op":          activation,
            "vpu_time_us":     round(vpu_r["vpu_time_us"], 4),
        }

    def simulate_workload(self, layers: list):
        results = []
        for layer in layers:
            name = layer.get("name", "layer")
            M, N, K = layer["M"], layer["N"], layer["K"]
            act = layer.get("activation", "identity")
            results.append(self.simulate_layer(name, M, N, K, act))
        return results
