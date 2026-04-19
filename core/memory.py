PRECISION_BYTES = {"int8": 1, "bf16": 2, "fp16": 2, "fp32": 4}


class MemorySimulator:
    """
    Models Unified Buffer (SRAM) + HBM bandwidth for a single GEMM layer.
    """
    def __init__(self, hbm_bandwidth_gbs: float, sram_size_kb: int,
                 sram_bandwidth_gbs: float, clock_mhz: float, precision: str = "bf16"):
        self.hbm_bw      = hbm_bandwidth_gbs * 1e9
        self.sram_bw     = sram_bandwidth_gbs * 1e9
        self.sram_bytes  = sram_size_kb * 1024
        self.clock_mhz   = clock_mhz
        self.elem_bytes  = PRECISION_BYTES.get(precision, 2)

    def simulate(self, M: int, N: int, K: int):
        bytes_ifmap  = M * K * self.elem_bytes
        bytes_filter = K * N * self.elem_bytes
        bytes_ofmap  = M * N * self.elem_bytes * 2

        total_bytes = bytes_ifmap + bytes_filter + bytes_ofmap
        in_sram     = total_bytes <= self.sram_bytes

        effective_bw = self.sram_bw if in_sram else self.hbm_bw
        mem_time_s   = total_bytes / effective_bw
        mem_cycles   = int(mem_time_s * self.clock_mhz * 1e6)

        return {
            "bytes_ifmap_kb":  bytes_ifmap  / 1024,
            "bytes_filter_kb": bytes_filter / 1024,
            "bytes_ofmap_kb":  bytes_ofmap  / 1024,
            "total_bytes_kb":  total_bytes  / 1024,
            "in_sram":         in_sram,
            "memory_cycles":   mem_cycles,
            "memory_time_us":  mem_time_s * 1e6,
        }
