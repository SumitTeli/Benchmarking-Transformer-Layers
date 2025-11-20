import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set a professional and readable style
sns.set(style="whitegrid", font_scale=1.2)

LAT_FILE = "latency_flops_results.csv"
MEM_FILE = "memory_bottlenecks.csv"

if not os.path.exists(LAT_FILE):
    raise FileNotFoundError("latency_flops_results.csv not found. Please run the benchmarking script first.")

lat = pd.read_csv(LAT_FILE)
device_name = lat["device"].iloc[0]
print("Generating the 5 most meaningful visualizations for:", device_name)

# --- DATA CLEANING AND PRE-CALCULATIONS ---
lat["seq_len"] = lat["seq_len"].astype(int)
lat["batch"] = lat["batch"].astype(int)
lat["model_size"] = lat["model_size"].astype(str)
lat["total_tokens"] = lat["seq_len"] * lat["batch"]

# -------------------------------------------------------------
# 1. FACET GRID — Latency (Model Size x Precision x Seq Len x Batch)
# The most comprehensive plot for absolute performance.
# -------------------------------------------------------------
print("Plot 1/5: Latency Facet Grid...")
plt.figure(figsize=(18, 10))
g = sns.catplot(
    data=lat,
    x="seq_len",
    y="mean_latency_sec",
    hue="batch",
    col="model_size",
    row="precision",
    kind="bar",
    palette="deep",
    errorbar=None,
    height=4, 
    aspect=1.2 
)
g.fig.suptitle(f"{device_name} — Mean Latency by All Variables", y=1.02, fontsize=16)
g.set_axis_labels("Sequence Length", "Mean Latency (s)")
g.set_titles(col_template="{col_name} Model", row_template="{row_name}")
g.tight_layout()
g.savefig(f"{device_name}_P1_latency_facet_all.png")
plt.close()

# -------------------------------------------------------------
# 2. LATENCY HEATMAP (Seq Len x Batch)
# Best for finding the absolute slowest / fastest combinations.
# We focus on the worst-case (Large, FP32) or a key case (Base, FP16).
# -------------------------------------------------------------
print("Plot 2/5: Latency Heatmap...")
heat_df = lat[(lat["model_size"] == "Base") & (lat["precision"] == "fp32")]
heat = heat_df.pivot_table(
    index="batch",
    columns="seq_len",
    values="mean_latency_sec"
)

plt.figure(figsize=(10,6))
sns.heatmap(
    heat,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=.5,
    cbar_kws={'label': 'Mean Latency (s)'}
)
plt.title(f"{device_name} — Latency Heatmap (Base Model, FP32)")
plt.xlabel("Sequence Length")
plt.ylabel("Batch Size")
plt.tight_layout()
plt.savefig(f"{device_name}_P2_latency_heatmap.png")
plt.close()

# -------------------------------------------------------------
# 3. TFLOPS HEATMAP
# Essential for analyzing computational efficiency.
# -------------------------------------------------------------
print("Plot 3/5: TFLOPS Heatmap...")
tflops_heat = lat.pivot_table(
    index=["model_size", "precision"],
    columns=["batch", "seq_len"],
    values="achieved_tflops"
)
# Create cleaner column labels
tflops_heat.columns = [f'B{b}_S{s}' for b, s in tflops_heat.columns]

plt.figure(figsize=(12, 7))
sns.heatmap(
    tflops_heat,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=.5,
    cbar_kws={'label': 'Achieved TFLOPS/s'}
)
plt.title(f"{device_name} — Achieved TFLOPS Heatmap (Efficiency)")
plt.ylabel("Model Size & Precision")
plt.xlabel("Configuration (Batch_Sequence)")
plt.tight_layout()
plt.savefig(f"{device_name}_P3_tflops_heatmap.png")
plt.close()

# -------------------------------------------------------------
# 4. THROUGHPUT SCALING (vs. Total Tokens)
# Shows how performance holds up as the total workload increases.
# -------------------------------------------------------------
print("Plot 4/5: Throughput Scaling...")
plt.figure(figsize=(10,6))
sns.lineplot(
    data=lat,
    x="total_tokens",
    y="throughput_tokens_sec",
    hue="model_size",
    style="precision",
    palette="husl",
    marker="o"
)
plt.title(f"{device_name} — Throughput Scaling by Total Tokens")
plt.xlabel("Total Tokens (Batch × Seq Len) [Log Scale]")
plt.ylabel("Throughput (Tokens/sec)")
plt.xscale('log') 
plt.legend(title="Configuration")
plt.tight_layout()
plt.savefig(f"{device_name}_P4_throughput_scaling.png")
plt.close()


# -------------------------------------------------------------
# 5. MEMORY UTILIZATION BAR CHART
# Crucial for identifying memory bottlenecks on CUDA/VRAM devices.
# -------------------------------------------------------------
print("Plot 5/5: Memory Utilization...")
if os.path.exists(MEM_FILE):
    mem = pd.read_csv(MEM_FILE)

    # --- Memory Data Cleaning ---
    if "utilization_pct" in mem.columns:
        mem = mem.copy()
        # Clean data that might contain 'N/A' or 'MiB' (only clean 'utilization_pct' here)
        mem["util"] = mem["utilization_pct"].astype(str).str.replace("%", "").str.strip()
        mem["util"] = pd.to_numeric(mem["util"], errors="coerce")
        mem.dropna(subset=['util'], inplace=True) # Remove non-CUDA/MPS entries

        if not mem.empty:
            plt.figure(figsize=(10,6))
            sns.barplot(
                data=mem,
                x="seq_len",
                y="util",
                hue="batch",
                palette="Set2",
                errorbar=None
            )
            plt.title(f"{device_name} — Peak Memory Utilization (%)")
            plt.xlabel("Sequence Length")
            plt.ylabel("Memory Utilization (%)")
            plt.tight_layout()
            plt.savefig(f"{device_name}_P5_memory_utilization.png")
            plt.close()
            print("✓ ALL 5 focused visualizations generated!")
        else:
            print("Skipping Memory Plot: Data in memory_bottlenecks.csv is not numeric (e.g., CPU/MPS only).")
    else:
        print(f"Skipping Memory Plot: '{MEM_FILE}' exists but missing 'utilization_pct' column (e.g., CPU/MPS only).")

else:
    print(f"Skipping Memory Plot: '{MEM_FILE}' not found. Generating 4 focused visualizations.")

print("\nAll required visualizations saved to the current directory.")