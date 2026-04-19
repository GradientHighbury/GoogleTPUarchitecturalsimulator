import configparser
import json
import os
from dataclasses import dataclass


@dataclass
class TPUConfig:
    name:                str   = "TPU_Custom"
    array_rows:          int   = 128
    array_cols:          int   = 128
    clock_mhz:           float = 940.0
    precision:           str   = "bf16"
    unified_buffer_kb:   int   = 32768
    accumulator_kb:      int   = 4096
    sram_bandwidth_gbs:  float = 8000.0
    hbm_bandwidth_gbs:   float = 900.0
    ici_bandwidth_gbs:   float = 400.0

    @classmethod
    def from_ini(cls, path: str):
        cp = configparser.ConfigParser()
        cp.read(path)
        s = cp["tpu"]
        return cls(
            name               = s.get("name", "TPU_Custom"),
            array_rows         = int(s.get("array_rows", 128)),
            array_cols         = int(s.get("array_cols", 128)),
            clock_mhz          = float(s.get("clock_mhz", 940)),
            precision          = s.get("precision", "bf16"),
            unified_buffer_kb  = int(s.get("unified_buffer_kb", 32768)),
            accumulator_kb     = int(s.get("accumulator_kb", 4096)),
            sram_bandwidth_gbs = float(s.get("sram_bandwidth_gbs", 8000)),
            hbm_bandwidth_gbs  = float(s.get("hbm_bandwidth_gbs", 900)),
            ici_bandwidth_gbs  = float(s.get("ici_bandwidth_gbs", 400)),
        )

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
