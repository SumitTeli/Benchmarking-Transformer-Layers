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
