class RooflineAnalyzer:
    """
    Determines arithmetic intensity and whether a layer is
    compute-bound or memory-bound relative to the HBM ridge point.
    """
    def __init__(self, peak_tops: float, hbm_bandwidth_gbs: float):
        self.peak_tops = peak_tops
        self.hbm_bw    = hbm_bandwidth_gbs
        self.ridge_point = (peak_tops * 1e12) / (hbm_bandwidth_gbs * 1e9)

    def analyze(self, M: int, N: int, K: int, precision_bytes: int = 2):
        flops = 2 * M * N * K
        bytes_accessed = (M * K + K * N + M * N) * precision_bytes
        arith_intensity = flops / bytes_accessed

        if arith_intensity >= self.ridge_point:
            bound = "COMPUTE"
            achievable_tops = self.peak_tops
        else:
            bound = "MEMORY"
            achievable_tops = arith_intensity * self.hbm_bw / 1e3

        return {
            "flops": flops,
            "bytes_accessed": bytes_accessed,
            "arith_intensity": arith_intensity,
            "ridge_point": self.ridge_point,
            "bound": bound,
            "achievable_tops": achievable_tops,
        }
