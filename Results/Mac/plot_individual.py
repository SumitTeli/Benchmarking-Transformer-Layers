import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

LAT_FILE = "latency_flops_results.csv"
MEM_FILE = "memory_bottlenecks.csv"

# ---------------------------------------
# LOAD CSV FILES IN CURRENT DIRECTORY
# ---------------------------------------
if not os.path.exists(LAT_FILE):
    raise FileNotFoundError("latency_flops_results.csv not found in this folder!")

if not os.path.exists(MEM_FILE):
    raise FileNotFoundError("memory_bottlenecks.csv not found in this folder!")

lat = pd.read_csv(LAT_FILE)
mem = pd.read_csv(MEM_FILE)

device_name = lat["device"].iloc[0]

print(f"Plotting individual results for: {device_name}")

# ---------------------------------------
# LATENCY vs SEQ LEN
# ---------------------------------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=lat,
    x="seq_len",
    y="mean_latency_sec",
    hue="batch",
    style="model_size",
    markers=True
)
plt.title(f"Latency vs Sequence Length – {device_name}")
plt.xlabel("Sequence Length")
plt.ylabel("Mean Latency (s)")
plt.savefig(f"{device_name}_latency.png")
plt.close()

# ---------------------------------------
# THROUGHPUT
# ---------------------------------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=lat,
    x="seq_len",
    y="throughput_tokens_sec",
    hue="batch",
    style="model_size",
    markers=True
)
plt.title(f"Throughput vs Sequence Length – {device_name}")
plt.xlabel("Sequence Length")
plt.ylabel("Tokens/sec")
plt.savefig(f"{device_name}_throughput.png")
plt.close()

# ---------------------------------------
# TFLOPs
# ---------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="model_size",
    y="achieved_tflops",
    hue="seq_len"
)
plt.title(f"Achieved TFLOPs – {device_name}")
plt.ylabel("TFLOPs/s")
plt.savefig(f"{device_name}_tflops.png")
plt.close()

# ---------------------------------------
# MEMORY UTILIZATION (GPU only)
# ---------------------------------------
if mem["total_vram"].iloc[0] != "Unified Memory":
    mem["util"] = mem["utilization_pct"].str.replace("%","").astype(float)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=mem,
        x="batch",
        y="util",
        hue="seq_len"
    )
    plt.title(f"Memory Utilization (%) – {device_name}")
    plt.ylabel("Percent of VRAM Used")
    plt.savefig(f"{device_name}_memory.png")
    plt.close()

print("✓ Individual plots generated.")
