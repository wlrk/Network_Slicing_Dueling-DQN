# plot_episode_from_csv.py

import os
import io
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_monitor_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file")
    # stable-baselines monitor: first line may be a commented header
    if lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    csv_text = "\n".join(lines).strip()
    if not csv_text:
        raise ValueError(f"{path}: no CSV content after header")
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip() for c in df.columns]
    df.insert(0, "episode", np.arange(1, len(df) + 1))
    return df

def episode_end_steps_from_monitor(df: pd.DataFrame) -> np.ndarray:
    if "l" not in df.columns:
        return np.array([], dtype=int)
    lvals = pd.to_numeric(df["l"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return np.cumsum(lvals).astype(int)

def try_load_loss_series_from_csv(csv_path: str):
    """
    Read loss time-series from CSV.

    Priority:
      A) 'episode' + 'loss'   -> (episodes, loss)
      B) ('step' or 'timesteps') + 'loss' -> (steps, loss)

    Returns:
      kind, x, y
      kind == "episode" -> x is episode index
      kind == "step"    -> x is step/timesteps
      On failure -> (None, None, None)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None, None
    cols = {c.strip().lower(): c for c in df.columns}
    def has(*names):
        return all(n in cols for n in names)

    if has("episode", "loss"):
        e = pd.to_numeric(df[cols["episode"]], errors="coerce").to_numpy()
        y = pd.to_numeric(df[cols["loss"]], errors="coerce").to_numpy()
        return "episode", e, y

    step_key = "step" if "step" in cols else ("timesteps" if "timesteps" in cols else None)
    if step_key and "loss" in cols:
        x = pd.to_numeric(df[cols[step_key]], errors="coerce").to_numpy()
        y = pd.to_numeric(df[cols["loss"]], errors="coerce").to_numpy()
        return "step", x, y

    return None, None, None

def find_default_loss_csv(outdir: str):
    # Prefer train-metrics.csv in the same output folder
    cand = os.path.join(outdir, "train-metrics.csv")
    return cand if os.path.exists(cand) else None

def align_step_series_to_episodes(ep_end_steps: np.ndarray, step_x: np.ndarray, step_y: np.ndarray) -> np.ndarray:
    """
    Align a step-based series (step_x, step_y) to episode end steps.
    For each episode end step, use the closest step <= end; else NaN.
    """
    if ep_end_steps.size == 0 or step_x is None or step_y is None or len(step_x) == 0:
        return np.full(ep_end_steps.shape, np.nan)
    order = np.argsort(step_x)
    sx = np.asarray(step_x)[order]
    sy = np.asarray(step_y)[order]
    out = np.full(ep_end_steps.shape, np.nan, dtype=float)
    j = 0
    for i, s in enumerate(ep_end_steps):
        while j + 1 < len(sx) and sx[j + 1] <= s:
            j += 1
        if sx[j] <= s:
            out[i] = sy[j]
    return out

def plot_episode_curve(ep, y, title, ylabel, outpath):
    plt.figure()
    plt.plot(ep, y)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_qoe(ep, qe=None, ql=None, outpath="episode_qoe.png"):
    plt.figure()
    have = False
    if qe is not None:
        plt.plot(ep, qe, label="Qe (eMBB)")
        have = True
    if ql is not None:
        plt.plot(ep, ql, label="Ql (URLLC)")
        have = True
    if not have:
        plt.close()
        return False
    plt.title("QoE per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Success ratio")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return True

def process_one_monitor(mon_path: str, loss_csv_glob: str = None):
    outdir = os.path.dirname(mon_path)
    dfm = read_monitor_csv(mon_path)
    ep = dfm["episode"].to_numpy()

    # 1) Reward
    if "r" in dfm.columns:
        r = pd.to_numeric(dfm["r"], errors="coerce").to_numpy()
        plot_episode_curve(ep, r, "Episode Reward", "Reward", os.path.join(outdir, "episode_reward.png"))

    # 2) Utility (if present)
    util = None
    if "utility" in dfm.columns:
        util = pd.to_numeric(dfm["utility"], errors="coerce").to_numpy()
        plot_episode_curve(ep, util, "Episode Utility", "Utility", os.path.join(outdir, "episode_utility.png"))

    # 3) QoE (if present)
    qe = pd.to_numeric(dfm["qe_embb"], errors="coerce").to_numpy() if "qe_embb" in dfm.columns else None
    ql = pd.to_numeric(dfm["ql_urllc"], errors="coerce").to_numpy() if "ql_urllc" in dfm.columns else None
    plot_qoe(ep, qe, ql, os.path.join(outdir, "episode_qoe.png"))

    # 4) Loss (optional)
    #   a) default: train-metrics.csv in same folder
    #   b) optional: use newest from --loss_csv glob
    loss_kind = None
    x_loss = y_loss = None
    chosen_loss_csv = None

    # optional glob first
    if loss_csv_glob:
        cands = glob.glob(loss_csv_glob, recursive=True)
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for cand in cands:
                kind, x, y = try_load_loss_series_from_csv(cand)
                if kind is not None:
                    loss_kind, x_loss, y_loss = kind, x, y
                    chosen_loss_csv = cand
                    break

    # fallback to default
    if loss_kind is None:
        default_loss = find_default_loss_csv(outdir)
        if default_loss:
            kind, x, y = try_load_loss_series_from_csv(default_loss)
            if kind is not None:
                loss_kind, x_loss, y_loss = kind, x, y
                chosen_loss_csv = default_loss

    if loss_kind is not None:
        if loss_kind == "episode":
            # already by episode; clip to common length
            n = min(len(ep), len(x_loss), len(y_loss))
            plot_episode_curve(ep[:n], np.asarray(y_loss[:n], dtype=float),
                               "Episode Loss", "Loss",
                               os.path.join(outdir, "episode_loss.png"))
        else:
            # step-based -> align to episode end steps
            ep_end = episode_end_steps_from_monitor(dfm)
            y_aligned = align_step_series_to_episodes(ep_end, x_loss, y_loss)
            plot_episode_curve(ep, y_aligned, "Episode Loss (aligned by end-step)", "Loss",
                               os.path.join(outdir, "episode_loss.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monitors", nargs="+", help="path(s) to monitor.csv")
    ap.add_argument("--loss_csv", type=str, default=None,
                    help="optional glob for a loss CSV (e.g., runs/**/output/train-metrics.csv)")
    args = ap.parse_args()

    for mon in args.monitors:
        try:
            process_one_monitor(mon, loss_csv_glob=args.loss_csv)
            print(f"[OK] {mon}")
        except Exception as e:
            print(f"[ERR] {mon}: {e}")

if __name__ == "__main__":
    main()
