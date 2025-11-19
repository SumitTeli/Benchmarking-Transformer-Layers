import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid", font_scale=1.3)

LAT_FILE = "latency_flops_results.csv"

if not os.path.exists(LAT_FILE):
    raise FileNotFoundError("latency_flops_results.csv not found.")

lat = pd.read_csv(LAT_FILE)
device_name = lat["device"].iloc[0]
print("Generating ALL clean visualizations for:", device_name)

# Clean small labels
lat["seq_len"] = lat["seq_len"].astype(int)
lat["batch"] = lat["batch"].astype(int)
lat["model_size"] = lat["model_size"].astype(str)

# -------------------------------------------------------------
# 1. SIMPLE BAR CHART — Latency (mean)
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="seq_len",
    y="mean_latency_sec",
    hue="batch",
    palette="Set2",
    errorbar=None
)
plt.title(f"{device_name} — Latency (Mean)")
plt.xlabel("Sequence Length")
plt.ylabel("Latency (s)")
plt.legend(title="Batch")
plt.tight_layout()
plt.savefig(f"{device_name}_latency_bar.png")
plt.close()

# -------------------------------------------------------------
# 2. CLUSTERED BAR CHART (seq_len × batch)
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="batch",
    y="mean_latency_sec",
    hue="seq_len",
    palette="Set1",
    errorbar=None
)
plt.title(f"{device_name} — Latency by Batch & Seq Len")
plt.xlabel("Batch Size")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig(f"{device_name}_latency_clustered.png")
plt.close()

# -------------------------------------------------------------
# 3. SMALL MULTIPLES (One Chart Per Batch)
# -------------------------------------------------------------
batches = sorted(lat["batch"].unique())
for b in batches:
    df = lat[lat["batch"] == b]

    plt.figure(figsize=(9,6))
    sns.barplot(
        data=df,
        x="seq_len",
        y="mean_latency_sec",
        color="skyblue",
        errorbar=None
    )
    plt.title(f"{device_name} — Latency (Batch {b})")
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    plt.savefig(f"{device_name}_latency_batch{b}.png")
    plt.close()

# -------------------------------------------------------------
# 4. HEATMAP (BEST SUMMARY VISUAL)
# -------------------------------------------------------------
heat = lat.pivot_table(
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
    linewidths=.5
)
plt.title(f"{device_name} — Latency Heatmap (s)")
plt.xlabel("Sequence Length")
plt.ylabel("Batch Size")
plt.tight_layout()
plt.savefig(f"{device_name}_latency_heatmap.png")
plt.close()

# -------------------------------------------------------------
# 5. THROUGHPUT BAR CHART
# -------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    data=lat,
    x="seq_len",
    y="throughput_tokens_sec",
    hue="batch",
    palette="Paired",
    errorbar=None
)
plt.title(f"{device_name} — Throughput vs Sequence")
plt.xlabel("Sequence Length")
plt.ylabel("Tokens/sec")
plt.tight_layout()
plt.savefig(f"{device_name}_throughput_bar.png")
plt.close()

# -------------------------------------------------------------
# 6. TFLOPS bar chart
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.barplot(
    data=lat,
    x="model_size",
    y="achieved_tflops",
    color="lightblue",
    errorbar=None
)
plt.title(f"{device_name} — Achieved TFLOPs")
plt.ylabel("TFLOPs/s")
plt.tight_layout()
plt.savefig(f"{device_name}_tflops.png")
plt.close()

print("✓ ALL clean and simple visualizations generated!")

# -------------------------------------------------------------
# 7. MEMORY VISUALIZATIONS (if memory_bottlenecks.csv present)
# -------------------------------------------------------------
MEM_FILE = "memory_bottlenecks.csv"

if os.path.exists(MEM_FILE):
    mem = pd.read_csv(MEM_FILE)

    # Assume the memory file belongs to this device/folder — clean fields
    if "utilization_pct" in mem.columns:
        mem = mem.copy()
        mem["util"] = mem["utilization_pct"].astype(str).str.replace("%", "").str.strip()
        mem["util"] = pd.to_numeric(mem["util"], errors="coerce")

    if "peak_allocated" in mem.columns:
        mem["peak_allocated_mb"] = (
            mem["peak_allocated"].astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # Memory utilization bar (seq_len × batch)
    if "util" in mem.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(
            data=mem,
            x="seq_len",
            y="util",
            hue="batch",
            palette="Set2",
            errorbar=None
        )
        plt.title(f"{device_name} — Memory Utilization (%)")
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Utilization (%)")
        plt.tight_layout()
        plt.savefig(f"{device_name}_memory_utilization.png")
        plt.close()

    # Peak allocated memory bar (MiB)
    if "peak_allocated_mb" in mem.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(
            data=mem,
            x="seq_len",
            y="peak_allocated_mb",
            hue="batch",
            palette="Paired",
            errorbar=None
        )
        plt.title(f"{device_name} — Peak Allocated Memory (MiB)")
        plt.xlabel("Sequence Length")
        plt.ylabel("Peak Allocated (MiB)")
        plt.tight_layout()
        plt.savefig(f"{device_name}_memory_peak_allocated.png")
        plt.close()

    # Heatmap of utilization (batch × seq_len)
    if "util" in mem.columns:
        try:
            heat_mem = mem.pivot_table(index="batch", columns="seq_len", values="util")
            plt.figure(figsize=(10,6))
            sns.heatmap(
                heat_mem,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                linewidths=.5
            )
            plt.title(f"{device_name} — Memory Utilization Heatmap (%)")
            plt.xlabel("Sequence Length")
            plt.ylabel("Batch Size")
            plt.tight_layout()
            plt.savefig(f"{device_name}_memory_heatmap.png")
            plt.close()
        except Exception:
            # If pivoting fails (e.g., non-numeric batches/seqs), skip heatmap
            pass

    print("✓ Memory visualizations generated (from memory_bottlenecks.csv).")
else:
    print("No memory_bottlenecks.csv found — skipping memory visualizations.")
