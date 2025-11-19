import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid", font_scale=1.2)

LAT_FILE = "latency_flops_results.csv"
MEM_FILE = "memory_bottlenecks.csv"

if not os.path.exists(LAT_FILE):
    raise FileNotFoundError("latency_flops_results.csv not found in this folder!")

lat = pd.read_csv(LAT_FILE)

device_name = lat["device"].iloc[0]
print(f"Generating individual plots for: {device_name}")

device_palette = {device_name: "blue"}  # single color for clarity

# ================================
# 1. Latency vs Sequence Length (3 subplots)
# ================================
batches = sorted(lat["batch"].unique())
fig, axs = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for ax, b in zip(axs, batches):
    df = lat[lat["batch"] == b]
    sns.lineplot(
        data=df,
        x="seq_len",
        y="mean_latency_sec",
        marker="o",
        ax=ax,
        color="blue"
    )
    ax.set_title(f"Latency (Batch={b})")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (s)")

plt.suptitle(f"{device_name} — Latency vs Sequence Length", fontsize=18)
plt.tight_layout()
plt.savefig(f"{device_name}_latency.png")
plt.close()


# ================================
# 2. Throughput (3 subplots)
# ================================
fig, axs = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for ax, b in zip(axs, batches):
    df = lat[lat["batch"] == b]
    sns.lineplot(
        data=df,
        x="seq_len",
        y="throughput_tokens_sec",
        marker="o",
        ax=ax,
        color="green"
    )
    ax.set_title(f"Throughput (Batch={b})")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens/sec")

plt.suptitle(f"{device_name} — Throughput vs Sequence Length", fontsize=18)
plt.tight_layout()
plt.savefig(f"{device_name}_throughput.png")
plt.close()


# ================================
# 3. Achieved TFLOPs
# ================================
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="model_size",
    y="achieved_tflops",
    palette="Blues_r"
)
plt.title(f"{device_name} — Achieved TFLOPs")
plt.ylabel("TFLOPs/s")
plt.savefig(f"{device_name}_tflops.png")
plt.close()


# ================================
# 4. FLOPs Efficiency (%)
# ================================
lat["est_tflops"] = lat["estimated_flops"] / 1e12
lat["flops_eff"] = (lat["achieved_tflops"] / lat["est_tflops"]) * 100

plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="model_size",
    y="flops_eff",
    palette="Purples"
)
plt.title(f"{device_name} — FLOPs Efficiency (%)")
plt.ylabel("Efficiency (%)")
plt.savefig(f"{device_name}_flops_eff.png")
plt.close()


# ================================
# 5. Latency p50 / p95 (jitter chart)
# ================================
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="seq_len",
    y="p95_sec",
    hue="model_size",
    palette="viridis"
)
plt.title(f"{device_name} — p95 Latency (Jitter)")
plt.ylabel("Latency (s)")
plt.savefig(f"{device_name}_p95.png")
plt.close()


# ================================
# 6. GPU Only — Memory Utilization
# ================================
if os.path.exists(MEM_FILE):
    mem = pd.read_csv(MEM_FILE)

    if mem["total_vram"].iloc[0] != "Unified Memory":
        mem["util"] = mem["utilization_pct"].str.replace("%","").astype(float)

        plt.figure(figsize=(10,6))
        sns.barplot(
            data=mem,
            x="seq_len",
            y="util",
            hue="batch",
            palette="magma"
        )
        plt.title(f"{device_name} — GPU Memory Utilization (%)")
        plt.ylabel("Percent VRAM Used")
        plt.savefig(f"{device_name}_memory.png")
        plt.close()

print("✓ Individual plots generated.")
