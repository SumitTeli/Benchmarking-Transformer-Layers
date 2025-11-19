import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys

sns.set(style="whitegrid", font_scale=1.2)

ROOT = sys.argv[1]

lat_list = []
mem_list = []

# -----------------------------
# Load data for each device
# -----------------------------
for folder in os.listdir(ROOT):
    path = os.path.join(ROOT, folder)
    if not os.path.isdir(path):
        continue

    lat_path = os.path.join(path, "latency_flops_results.csv")
    mem_path = os.path.join(path, "memory_bottlenecks.csv")

    if not os.path.exists(lat_path):
        continue

    lat = pd.read_csv(lat_path)
    lat["device"] = folder
    lat_list.append(lat)

    mem = pd.read_csv(mem_path)
    mem["device"] = folder
    mem_list.append(mem)

lat_all = pd.concat(lat_list)
mem_all = pd.concat(mem_list)

# Device color scheme
device_palette = {
    "cpu": "red",
    "gpu-1650": "green",
    "gpu-5060": "blue",
    "mac": "orange"
}

# ----------------------------------------------------
# 1. LATENCY vs SEQ_LEN (3 SUBPLOTS BY BATCH SIZE)
# ----------------------------------------------------
batches = sorted(lat_all["batch"].unique())
fig, axs = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for ax, b in zip(axs, batches):
    df = lat_all[lat_all["batch"] == b]
    sns.lineplot(
        data=df,
        x="seq_len",
        y="mean_latency_sec",
        hue="device",
        palette=device_palette,
        marker="o",
        ax=ax
    )
    ax.set_title(f"Latency (Batch={b})")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (s)")

plt.suptitle("Global Latency vs Sequence Length", fontsize=18)
plt.tight_layout()
plt.savefig("global_latency.png")
plt.close()

# ----------------------------------------------------
# 2. THROUGHPUT (3 SUBPLOTS)
# ----------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for ax, b in zip(axs, batches):
    df = lat_all[lat_all["batch"] == b]
    sns.lineplot(
        data=df,
        x="seq_len",
        y="throughput_tokens_sec",
        hue="device",
        palette=device_palette,
        marker="o",
        ax=ax
    )
    ax.set_title(f"Throughput (Batch={b})")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens/sec")

plt.suptitle("Global Throughput Comparison", fontsize=18)
plt.tight_layout()
plt.savefig("global_throughput.png")
plt.close()

# ----------------------------------------------------
# 3. Achieved TFLOPs (BAR CHART)
# ----------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat_all,
    x="device",
    y="achieved_tflops",
    hue="model_size",
    palette=device_palette
)
plt.title("Achieved TFLOPs by Device")
plt.ylabel("TFLOPs/s")
plt.savefig("global_tflops.png")
plt.close()

# ----------------------------------------------------
# 4. p95 Latency (BAR CHART)
# ----------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat_all,
    x="device",
    y="p95_sec",
    hue="seq_len",
    palette=device_palette
)
plt.title("p95 Latency Comparison")
plt.ylabel("Latency (s)")
plt.savefig("global_p95.png")
plt.close()

# ----------------------------------------------------
# 5. FLOPs Efficiency (%)
# ----------------------------------------------------
lat_all["est_tflops"] = lat_all["estimated_flops"] / 1e12
lat_all["flops_eff"] = (lat_all["achieved_tflops"] / lat_all["est_tflops"]) * 100

plt.figure(figsize=(10,6))
sns.barplot(
    data=lat_all,
    x="device",
    y="flops_eff",
    hue="model_size",
    palette=device_palette
)
plt.title("FLOPs Efficiency (%) by Device")
plt.ylabel("Efficiency (%)")
plt.savefig("global_flops_eff.png")
plt.close()

# ----------------------------------------------------
# 6. MEMORY UTILIZATION (ONLY GPUs)
# ----------------------------------------------------
gpu_mem = mem_all[mem_all["total_vram"] != "Unified Memory"].copy()
if len(gpu_mem) > 0:
    gpu_mem["util"] = gpu_mem["utilization_pct"].str.replace("%","").astype(float)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=gpu_mem,
        x="device",
        y="util",
        hue="seq_len",
        palette=device_palette
    )
    plt.title("GPU Memory Utilization (%)")
    plt.ylabel("Percent VRAM Used")
    plt.savefig("global_memory.png")
    plt.close()
