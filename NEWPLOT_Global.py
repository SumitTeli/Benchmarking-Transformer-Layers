import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.3)

ROOT = "results"

lat_list = []
mem_list = []

# -------------------------------------------------------------------
# 1. READ AND CONCATENATE ALL DEVICE FOLDERS
# -------------------------------------------------------------------
if not os.path.isdir(ROOT):
    raise FileNotFoundError(f"Results directory '{ROOT}' not found. Please create it and place device folders inside.")

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

if not lat_list:
    raise ValueError(f"No valid 'latency_flops_results.csv' files found in any subfolder of '{ROOT}'.")

# MERGED DATA
lat = pd.concat(lat_list, ignore_index=True)
mem = pd.concat(mem_list, ignore_index=True) if mem_list else None

# Clean
lat["seq_len"] = lat["seq_len"].astype(int)
lat["batch"] = lat["batch"].astype(int)
lat["model_size"] = lat["model_size"].astype(str)
print(f"Successfully loaded data from {lat['device'].nunique()} devices.")

# -------------------------------------------------------------------
# 2. GLOBAL LATENCY COMPARISON (BAR CHART: Seq Len as X)
# -------------------------------------------------------------------
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat,
    x="seq_len",
    y="mean_latency_sec",
    hue="device",
    errorbar=None
)
plt.title("Global Latency Comparison (Mean Latency by Seq Len)")
plt.xlabel("Sequence Length")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig("global_P2_latency_compare_seq.png")
plt.close()


# -------------------------------------------------------------------
# 3. GLOBAL THROUGHPUT COMPARISON (BAR CHART: Seq Len as X)
# -------------------------------------------------------------------
plt.figure(figsize=(12,7))
sns.barplot(
    data=lat,
    x="seq_len",
    y="throughput_tokens_sec",
    hue="device",
    errorbar=None
)
plt.title("Global Throughput Comparison (Tokens/sec by Seq Len)")
plt.xlabel("Sequence Length")
plt.ylabel("Tokens/sec")
plt.tight_layout()
plt.savefig("global_P3_throughput_compare_seq.png")
plt.close()


# -------------------------------------------------------------------
# 4. GLOBAL TFLOPs Comparison (FACET GRID) - BEST EFFICIENCY PLOT
# -------------------------------------------------------------------
print("Plot 4/7: TFLOPs Facet Grid...")
g = sns.catplot(
    data=lat,
    x="model_size",
    y="achieved_tflops",
    hue="device",
    col="precision", # Separate columns by Precision
    kind="bar",
    palette="tab10",
    errorbar=None,
    height=6,
    aspect=0.8
)
g.fig.suptitle("Global TFLOPs Comparison by Precision and Model Size", y=1.02, fontsize=16)
g.set_axis_labels("Model Size", "Achieved TFLOPs/s")
g.set_titles(col_template="{col_name}")
g.tight_layout()
plt.savefig("global_P4_tflops_facet_compare.png")
plt.close()


# -------------------------------------------------------------------
# 5. GLOBAL LATENCY HEATMAP (device × seq_len)
# Filter to a consistent precision for a clean comparison (FP32 is safest)
# -------------------------------------------------------------------
print("Plot 5/7: Latency Heatmap (FP32)...")
heat_latency_df = lat[lat['precision'] == 'fp32'] 
heat_latency = heat_latency_df.pivot_table(
    index="device",
    columns="seq_len",
    values="mean_latency_sec"
)

plt.figure(figsize=(12,6))
sns.heatmap(
    heat_latency,
    annot=True,
    cmap="YlOrRd",
    fmt=".2f",
    cbar_kws={'label': 'Mean Latency (s)'}
)
plt.title("Latency Heatmap (Device × Seq Len) [FP32 Only]")
plt.tight_layout()
plt.savefig("global_P5_latency_heatmap_fp32.png")
plt.close()


# -------------------------------------------------------------------
# 6. GLOBAL THROUGHPUT HEATMAP
# Filter to a consistent precision for a clean comparison (FP32 is safest)
# -------------------------------------------------------------------
print("Plot 6/7: Throughput Heatmap (FP32)...")
heat_tp_df = lat[lat['precision'] == 'fp32'] 
heat_tp = heat_tp_df.pivot_table(
    index="device",
    columns="seq_len",
    values="throughput_tokens_sec"
)

plt.figure(figsize=(12,6))
sns.heatmap(
    heat_tp,
    annot=True,
    cmap="YlGnBu",
    fmt=".0f",
    cbar_kws={'label': 'Tokens/sec'}
)
plt.title("Throughput Heatmap (Device × Seq Len) [FP32 Only]")
plt.tight_layout()
plt.savefig("global_P6_throughput_heatmap_fp32.png")
plt.close()


# -------------------------------------------------------------------
# 7. VRAM UTILIZATION (GPU Only)
# -------------------------------------------------------------------
print("Plot 7/7: VRAM Utilization...")
if mem is not None:
    # Filter out non-GPU devices (those with 'Unified Memory')
    mem_gpu = mem[mem["total_vram"] != "Unified Memory"].copy() 

    if not mem_gpu.empty:
        # Clean 'utilization_pct' column to a float
        mem_gpu["util_pct"] = mem_gpu["utilization_pct"].astype(str).str.replace("%","").str.strip().astype(float)
        
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
        plt.savefig("global_P7_vram_utilization.png")
        plt.close()
        print("✓ ALL 7 global comparison plots generated successfully!")
    else:
        print("Skipping VRAM Utilization Plot: No dedicated VRAM devices found in memory data.")
else:
    print("Skipping VRAM Utilization Plot: memory_bottlenecks.csv files not found.")

print("\nAll global comparison plots saved to the current directory.")