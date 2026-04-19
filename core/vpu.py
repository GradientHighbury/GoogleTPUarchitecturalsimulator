class VPUSimulator:
    """
    Models the Vector Processing Unit for elementwise ops.
    """
    THROUGHPUT_ELEM_PER_CYCLE = {
        "relu":      128,
        "gelu":      32,
        "layernorm": 16,
        "softmax":   32,
        "add":       128,
        "identity":  256,
    }

    def __init__(self, clock_mhz: float, precision_bytes: int = 2):
        self.clock_mhz  = clock_mhz
        self.prec_bytes = precision_bytes

    def simulate(self, op: str, num_elements: int):
        throughput = self.THROUGHPUT_ELEM_PER_CYCLE.get(op.lower(), 64)
        cycles     = num_elements / throughput
        time_us    = cycles / (self.clock_mhz * 1e6) * 1e6
        bw_bytes   = num_elements * self.prec_bytes * 2
        return {
            "vpu_op":      op,
            "elements":    num_elements,
            "vpu_cycles":  cycles,
            "vpu_time_us": time_us,
            "vpu_bw_bytes":bw_bytes,
        }
