import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------
# DEVICE DETECTION
# -----------------------------------------------------------
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = detect_device()
print(f"\nRunning on: {device}\n")

# -----------------------------------------------------------
# VIT TRANSFORMER-ENCODER LAYER
# -----------------------------------------------------------
class ViTEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

    def forward(self, x):
        return self.layer(x)

# -----------------------------------------------------------
# FLOPs ESTIMATION (Attention + FFN)
# -----------------------------------------------------------
def estimate_flops(seq_len, d_model, ff_dim):
    attn_proj = 4 * seq_len * (d_model * d_model)
    attn_scores = 2 * (seq_len * seq_len * d_model)
    ffn = 2 * seq_len * (d_model * ff_dim)
    return attn_proj + attn_scores + ffn

# -----------------------------------------------------------
# MEMORY MEASUREMENT
# -----------------------------------------------------------
def measure_memory():
    if device.type != "cuda":
        return {
            "total_vram": "Unified Memory",
            "peak_allocated": "N/A",
            "peak_reserved": "N/A",
            "utilization_pct": "N/A",
        }

    torch.cuda.synchronize()

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    utilization = (peak_allocated / total_vram) * 100

    return {
        "total_vram": f"{total_vram:.1f} MiB",
        "peak_allocated": f"{peak_allocated:.1f} MiB",
        "peak_reserved": f"{peak_reserved:.1f} MiB",
        "utilization_pct": f"{utilization:.1f} %",
    }

# PRECISION MODES
def get_precision_modes():
    if device.type == "cuda":
        return ["fp32", "fp16"]
    else:
        return ["fp32"]

precision_modes = get_precision_modes()

# -----------------------------------------------------------
# BENCHMARK CONFIG
# -----------------------------------------------------------
model_sizes = [
    {"name": "Small", "d": 384, "h": 6, "ff": 1536},
    {"name": "Base", "d": 768, "h": 12, "ff": 3072},
    {"name": "Large", "d": 1024, "h": 16, "ff": 4096},
]

sequence_lengths = [196, 576, 1024]
batch_sizes = [1, 8, 32]

num_warmup = 3
num_iters = 10

lat_results = []
mem_results = []

# -----------------------------------------------------------
# MAIN BENCHMARK LOOP
# -----------------------------------------------------------
for cfg in model_sizes:
    for precision in precision_modes:

        layer = ViTEncoderLayer(cfg["d"], cfg["h"], cfg["ff"]).to(device)
        if precision == "fp16":
            layer = layer.half()

        for seq_len in sequence_lengths:
            for batch in batch_sizes:

                x = torch.randn(batch, seq_len, cfg["d"]).to(device)
                if precision == "fp16":
                    x = x.half()

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                # Warmup
                for _ in range(num_warmup):
                    _ = layer(x)

                latencies = []
                total_tokens = seq_len * batch

                # Timed iterations
                for _ in tqdm(range(num_iters), desc=f"{cfg['name']} B{batch} S{seq_len}"):
                    start = time.time()
                    _ = layer(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if device.type == "mps":
                        torch.mps.synchronize()
                    end = time.time()
                    latencies.append(end - start)

                avg_lat = float(np.mean(latencies))
                p50_lat = float(np.percentile(latencies, 50))
                p95_lat = float(np.percentile(latencies, 95))
                wall_time = float(np.sum(latencies))
                throughput_tokens = total_tokens / avg_lat

                # FLOPs
                total_flops = estimate_flops(seq_len, cfg["d"], cfg["ff"]) * batch
                achieved_tflops = (total_flops / avg_lat) / 1e12

                # Memory
                mem = measure_memory()
                mem_results.append({
                    "device": str(device),
                    "model": cfg["name"],
                    "precision": precision,
                    "seq_len": seq_len,
                    "batch": batch,
                    **mem
                })

                # Save latency + FLOPs
                lat_results.append({
                    "device": str(device),
                    "model_size": cfg["name"],
                    "precision": precision,
                    "seq_len": seq_len,
                    "batch": batch,
                    "mean_latency_sec": round(avg_lat, 6),
                    "p50_sec": round(p50_lat, 6),
                    "p95_sec": round(p95_lat, 6),
                    "throughput_tokens_sec": round(throughput_tokens, 2),
                    "wall_time_sec": round(wall_time, 4),
                    "estimated_flops": int(total_flops),
                    "achieved_tflops": round(achieved_tflops, 4),
                })

# SAVE RESULTS
lat_df = pd.DataFrame(lat_results)
lat_df.to_csv("latency_flops_results.csv", index=False)

mem_df = pd.DataFrame(mem_results)
mem_df.to_csv("memory_bottlenecks.csv", index=False)

print("\n✔ Saved:")
print("latency_flops_results.csv")
print("memory_bottlenecks.csv")
