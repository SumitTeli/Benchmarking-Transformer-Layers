import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sns.set(style="whitegrid")

ROOT = sys.argv[1]

lat_list = []
mem_list = []

# ---------------------------------------
# Load all device CSVs
# ---------------------------------------
for folder in os.listdir(ROOT):
    path = os.path.join(ROOT, folder)
    if not os.path.isdir(path):
        continue

    lat_path = os.path.join(path, "latency_flops_results.csv")
    mem_path = os.path.join(path, "memory_bottlenecks.csv")

    if not os.path.exists(lat_path):
        continue

    lat = pd.read_csv(lat_path)
    lat["device_label"] = folder
    lat_list.append(lat)

    mem = pd.read_csv(mem_path)
    mem["device_label"] = folder
    mem_list.append(mem)

lat_all = pd.concat(lat_list, ignore_index=True)
mem_all = pd.concat(mem_list, ignore_index=True)

print("Loaded devices:", lat_all["device_label"].unique())

# ==============================================================
#     1. LATENCY VS SEQ LEN
# ==============================================================
plt.figure(figsize=(12,7))
sns.lineplot(
    data=lat_all,
    x="seq_len",
    y="mean_latency_sec",
    hue="device_label",
    style="batch",
    markers=True
)
plt.title("Global Latency vs Sequence Length")
plt.savefig("global_latency_vs_seq.png")
plt.close()

# ==============================================================
#     2. LATENCY VS BATCH SIZE
# ==============================================================
plt.figure(figsize=(12,7))
sns.lineplot(
    data=lat_all,
    x="batch",
    y="mean_latency_sec",
    hue="device_label",
    style="seq_len",
    markers=True
)
plt.title("Global Latency vs Batch Size")
plt.savefig("global_latency_vs_batch.png")
plt.close()

# ==============================================================
#     3. p50/p95 Comparison
# ==============================================================
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat_all,
    x="device_label",
    y="p95_sec",
    hue="model_size"
)
plt.title("Global p95 Latency")
plt.savefig("global_p95.png")
plt.close()

# ==============================================================
#     4. THROUGHPUT
# ==============================================================
plt.figure(figsize=(12,7))
sns.lineplot(
    data=lat_all,
    x="seq_len",
    y="throughput_tokens_sec",
    hue="device_label",
    style="batch"
)
plt.title("Global Throughput Comparison")
plt.savefig("global_throughput.png")
plt.close()

# ==============================================================
#     5. Achieved TFLOPs
# ==============================================================
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat_all,
    x="device_label",
    y="achieved_tflops",
    hue="model_size"
)
plt.title("Global Achieved TFLOPs")
plt.savefig("global_tflops.png")
plt.close()

# ==============================================================
#     6. FLOPs Efficiency %
# ==============================================================
lat_all["flops_eff"] = (lat_all["achieved_tflops"] / (lat_all["estimated_flops"] / 1e12)) * 100

plt.figure(figsize=(12,7))
sns.barplot(
    data=lat_all,
    x="device_label",
    y="flops_eff",
    hue="model_size"
)
plt.title("FLOPs Efficiency (%)")
plt.savefig("global_flops_efficiency.png")
plt.close()

# ==============================================================
#     7. MEMORY UTILIZATION (GPU ONLY)
# ==============================================================
gpu_mem = mem_all[mem_all["total_vram"] != "Unified Memory"].copy()
if len(gpu_mem) > 0:
    gpu_mem["util"] = gpu_mem["utilization_pct"].str.replace("%","").astype(float)

    plt.figure(figsize=(12,7))
    sns.barplot(
        data=gpu_mem,
        x="device_label",
        y="util",
        hue="seq_len"
    )
    plt.title("Global Memory Utilization (%) – GPUs Only")
    plt.savefig("global_memory_usage.png")
    plt.close()

print("✓ All global comparison plots generated.")
