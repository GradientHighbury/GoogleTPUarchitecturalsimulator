import math


class MXUSimulator:
    """
    Systolic-array Matrix Multiply Unit (weight-stationary dataflow).
    """
    def __init__(self, array_rows: int, array_cols: int, clock_mhz: float, precision: str = "bf16"):
        self.rows      = array_rows
        self.cols      = array_cols
        self.clock_mhz = clock_mhz
        self.precision = precision
        self.peak_tops = 2 * array_rows * array_cols * clock_mhz * 1e6 / 1e12

    def simulate_gemm(self, M: int, N: int, K: int):
        m_tiles = math.ceil(M / self.rows)
        n_tiles = math.ceil(N / self.cols)
        k_tiles = math.ceil(K / self.cols)

        fill_drain     = self.rows + self.cols - 1
        compute_cycles = m_tiles * n_tiles * (k_tiles * self.cols + fill_drain)

        useful_macs        = M * N * K
        peak_macs_in_cycles = compute_cycles * self.rows * self.cols
        mxu_util           = useful_macs / peak_macs_in_cycles if peak_macs_in_cycles > 0 else 0.0

        time_s         = compute_cycles / (self.clock_mhz * 1e6)
        compute_tflops = (2 * useful_macs) / (time_s * 1e12) if time_s > 0 else 0.0

        return {
            "compute_cycles":  compute_cycles,
            "useful_macs":     useful_macs,
            "mxu_utilization": mxu_util,
            "compute_tflops":  compute_tflops,
            "time_us":         time_s * 1e6,
        }
