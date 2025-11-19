import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.3)

ROOT = "results"

lat_list = []
mem_list = []

# -------------------------------------------------------------------
# 1. READ ALL DEVICE FOLDERS
# -------------------------------------------------------------------
for device_folder in os.listdir(ROOT):
    path = os.path.join(ROOT, device_folder)
    if not os.path.isdir(path):
        continue

    lat_file = os.path.join(path, "latency_flops_results.csv")
    mem_file = os.path.join(path, "memory_bottlenecks.csv")

    if os.path.exists(lat_file):
        df_lat = pd.read_csv(lat_file)
        df_lat["device"] = device_folder
        lat_list.append(df_lat)

    if os.path.exists(mem_file):
        df_mem = pd.read_csv(mem_file)
        df_mem["device"] = device_folder
        mem_list.append(df_mem)

# MERGED DATA
lat = pd.concat(lat_list, ignore_index=True)
mem = pd.concat(mem_list, ignore_index=True) if mem_list else None

# Clean
lat["seq_len"] = lat["seq_len"].astype(int)
lat["batch"] = lat["batch"].astype(int)
lat["model_size"] = lat["model_size"].astype(str)


# -------------------------------------------------------------------
# 2. GLOBAL LATENCY COMPARISON (BAR CHART)
# -------------------------------------------------------------------
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat,
    x="seq_len",
    y="mean_latency_sec",
    hue="device",
    errorbar=None
)
plt.title("Global Latency Comparison (Mean Latency)")
plt.xlabel("Sequence Length")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig("global_latency_compare.png")
plt.close()


# -------------------------------------------------------------------
# 3. GLOBAL THROUGHPUT COMPARISON
# -------------------------------------------------------------------
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat,
    x="seq_len",
    y="throughput_tokens_sec",
    hue="device",
    errorbar=None
)
plt.title("Global Throughput Comparison")
plt.xlabel("Sequence Length")
plt.ylabel("Tokens/sec")
plt.tight_layout()
plt.savefig("global_throughput_compare.png")
plt.close()


# -------------------------------------------------------------------
# 4. GLOBAL TFLOPs Comparison (Bar Chart)
# -------------------------------------------------------------------
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat,
    x="model_size",
    y="achieved_tflops",
    hue="device",
    errorbar=None
)
plt.title("Global TFLOPs Comparison Across Devices")
plt.ylabel("TFLOPs/s")
plt.tight_layout()
plt.savefig("global_tflops_compare.png")
plt.close()


# -------------------------------------------------------------------
# 5. GLOBAL LATENCY HEATMAP (device × seq_len)
# -------------------------------------------------------------------
heat_latency = lat.pivot_table(
    index="device",
    columns="seq_len",
    values="mean_latency_sec"
)

plt.figure(figsize=(12,6))
sns.heatmap(
    heat_latency,
    annot=True,
    cmap="YlOrRd",
    fmt=".2f"
)
plt.title("Latency Heatmap (Device × Seq Len)")
plt.tight_layout()
plt.savefig("global_latency_heatmap.png")
plt.close()


# -------------------------------------------------------------------
# 6. GLOBAL THROUGHPUT HEATMAP
# -------------------------------------------------------------------
heat_tp = lat.pivot_table(
    index="device",
    columns="seq_len",
    values="throughput_tokens_sec"
)

plt.figure(figsize=(12,6))
sns.heatmap(
    heat_tp,
    annot=True,
    cmap="YlGnBu",
    fmt=".0f"
)
plt.title("Throughput Heatmap (Device × Seq Len)")
plt.tight_layout()
plt.savefig("global_throughput_heatmap.png")
plt.close()


# -------------------------------------------------------------------
# 7. VRAM UTILIZATION (GPU Only)
# -------------------------------------------------------------------
if mem is not None:
    mem_gpu = mem[mem["total_vram"] != "Unified Memory"]

    if not mem_gpu.empty:
        mem_gpu["util_pct"] = mem_gpu["utilization_pct"].str.replace("%","").astype(float)

        plt.figure(figsize=(12,7))
        sns.barplot(
            data=mem_gpu,
            x="device",
            y="util_pct",
            hue="seq_len",
            errorbar=None
        )
        plt.title("GPU VRAM Utilization (%) Across Devices")
        plt.ylabel("VRAM Used (%)")
        plt.tight_layout()
        plt.savefig("global_vram_utilization.png")
        plt.close()

print("✓ GLOBAL COMPARISON PLOTS GENERATED SUCCESSFULLY!")
