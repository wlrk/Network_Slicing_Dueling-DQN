# plot_from_console.py — unified training visualizer (auto-detect SE units)
import os
import re
import glob
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- console parsing ----------
def read_text_auto(path):
    for enc in ("utf-16", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", errors="ignore") as f:
        return f.read()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _plot_curve(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def _plot_twin(x, y1, y2, y1_label, y2_label, title, xlabel, outpath):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def find_latest_csv(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def parse_train_metrics_csv(path):
    """
    Read train-metrics.csv into dict(list).
    Required: timesteps, reward
    Optional: se (bps/Hz), se_mbps, utility, qe_embb, ql_urllc, etc.
    """
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header is None:
            return out
        keys = [h.strip() for h in header]
        for k in keys:
            out[k] = []
        for row in rd:
            if not row:
                continue
            for k, v in zip(keys, row):
                v = v.strip()
                if v == "":
                    out[k].append(np.nan)
                    continue
                try:
                    out[k].append(float(v))
                except ValueError:
                    out[k].append(v)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs",
                        help="directory containing console logs or csv metrics")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="directory to save plots")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    csv_path = find_latest_csv([
        os.path.join(args.logdir, "**", "train-metrics.csv"),
        os.path.join(args.logdir, "train-metrics.csv"),
    ])

    if csv_path:
        try:
            df = parse_train_metrics_csv(csv_path)
            xs = df.get("timesteps", [])
            has_se_mbps = ("se_mbps" in df) and any(np.isfinite(x) for x in df["se_mbps"])
            has_se = ("se" in df) and any(np.isfinite(x) for x in df["se"])

            if has_se_mbps:
                se_series = df["se_mbps"]
                se_label = "SE (Mbps)"
            elif has_se:
                se_series = df["se"]
                se_label = "SE (bps/Hz)"
            else:
                se_series = None
                se_label = "SE"

            if "reward" in df and xs:
                _plot_curve(xs, df["reward"],
                            "Episode Reward vs Iterations",
                            "Iterations (timesteps)", "Reward",
                            os.path.join(outdir, "reward_vs_iterations.png"))

            if "qe_embb" in df and "ql_urllc" in df and xs:
                plt.figure()
                plt.plot(xs, df["qe_embb"], label="Qe (eMBB)")
                plt.plot(xs, df["ql_urllc"], label="Ql (URLLC)")
                plt.title("QoE (Success Ratio) vs Iterations")
                plt.xlabel("Iterations (timesteps)")
                plt.ylabel("Success ratio")
                plt.ylim(0.0, 1.05)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, "qoe_vs_iterations.png"), dpi=150)
                plt.close()

            if se_series is not None and xs:
                _plot_curve(xs, se_series,
                            "Spectral Efficiency vs Iterations",
                            "Iterations (timesteps)", se_label,
                            os.path.join(outdir, "se_vs_iterations.png"))

            if "utility" in df and se_series is not None and xs:
                _plot_twin(xs, df["utility"], se_series,
                           "Utility", se_label,
                           "Utility & SE vs Iterations",
                           "Iterations (timesteps)",
                           os.path.join(outdir, "utility_se_vs_iterations.png"))
        except Exception as e:
            print(f"[warn] failed to plot {csv_path}: {e}")

    print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    main()
