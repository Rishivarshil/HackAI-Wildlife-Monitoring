import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Adjust to where your CSV files are
DATA_FOLDER = "./monitor"
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files found.")

print(f"Found {len(csv_files)} CSV files")

runs = []

# ----------------------------
# Normalize headers
# ----------------------------
def normalize_columns(df):
    rename_map = {}

    for col in df.columns:
        c = col.lower()

        if c in ["elapsed_s", "elapsed_time_s"]:
            rename_map[col] = "elapsed_s"
        elif c in ["cpu_percent_total", "cpu_percent"]:
            rename_map[col] = "cpu_percent"
        elif c in ["proc_rss_mb", "process_rss_mb"]:
            rename_map[col] = "proc_rss_mb"
        elif c in ["sys_mem_percent", "system_memory_percent"]:
            rename_map[col] = "sys_mem_percent"
        elif c == "system_memory_used_mb":
            rename_map[col] = "sys_mem_used_mb"
        elif c in ["disk_used_gb"]:
            rename_map[col] = "disk_used_gb"
        elif c in ["disk_free_gb"]:
            rename_map[col] = "disk_free_gb"

    df = df.rename(columns=rename_map)
    return df


# ----------------------------
# Load + Normalize
# ----------------------------
for file in csv_files:
    df = pd.read_csv(file)
    df = normalize_columns(df)
    run_name = os.path.basename(file)
    df["run_name"] = os.path.basename(file)

    # Estimate power usage (Watts)
    if "osc" in run_name.lower():
        idle_w = 50.0
        max_w = 200.0
    else:
        idle_w = 3.0
        max_w = 12.0

    if "cpu_percent" in df.columns:
        df["power_watts_est"] = idle_w + (df["cpu_percent"] / 100.0) * (max_w - idle_w)
    else:
        df["power_watts_est"] = None

    runs.append(df)

# ----------------------------
# Plot helper
# ----------------------------
def plot_metric(metric, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for df in runs:
        if metric in df.columns:
            plt.plot(df["elapsed_s"], df[metric], label=df["run_name"].iloc[0])

    plt.xlabel("Elapsed Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(DATA_FOLDER, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {filename}")


# ----------------------------
# Generate plots
# ----------------------------
plot_metric("cpu_percent", "CPU Usage (%)", "cpu_usage.png")
plot_metric("proc_rss_mb", "Process RSS Memory (MB)", "process_memory.png")
plot_metric("sys_mem_percent", "System Memory (%)", "system_memory.png")
plot_metric("power_watts_est", "Estimated Power Usage (Watts)", "estimated_power.png")

# ----------------------------
# Summary Comparison
# ----------------------------
summary_rows = []

for df in runs:
    run_name = df["run_name"].iloc[0]

    summary = {
        "run": run_name,
        "runtime_s": df["elapsed_s"].max(),
        "avg_cpu_%": df["cpu_percent"].mean() if "cpu_percent" in df else None,
        "peak_cpu_%": df["cpu_percent"].max() if "cpu_percent" in df else None,
        "peak_mem_mb": df["proc_rss_mb"].max() if "proc_rss_mb" in df else None,
        "avg_power_w": df["power_watts_est"].mean() if "power_watts_est" in df else None,
        "peak_power_w": df["power_watts_est"].max() if "power_watts_est" in df else None,
    }

    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("run_summary.csv", index=False)

print("\nSummary saved to run_summary.csv")
print(summary_df)