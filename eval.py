# eval.py — eval with reservation-hold logs 
import os, csv, datetime, argparse
import pandas as pd
import config
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from envs.network_env import NetworkSlicingEnv as NS_Env

RUNS_ROOT = r"C:\Users\network_slicing_dqnpaper\runs"

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _load_trace_df(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-16", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    try:
        return pd.read_csv(path)
    except Exception:
        raise RuntimeError(f"[EVAL] Failed to read TRACE_FILE={path}; last_err={last_err}")

def _snap_config(out_dir: str):
    snap_path = os.path.join(out_dir, "eval_config_snapshot.txt")
    keys = [
        "DEMAND_MODE","TRACE_FILE","SLOT_MS","EP_LEN","TOTAL_BW_MHZ","BW_RES_MHZ",
        "EMBB_MIN_BW_MHZ","ENFORCE_DELAY","EMBB_MAX_DELAY_MS","URLLC_MAX_DELAY_MS",
        "EMBB_ARRIVAL_DIST","EMBB_ARRIVAL_MEAN_MS","EMBB_ARRIVAL_MAX_MS","EMBB_PARETO_ALPHA",
        "EMBB_PKT_DIST","EMBB_PKT_MEAN_BYTE","EMBB_PKT_LOGN_SIGMA","EMBB_PKT_MIN_BYTE","EMBB_PKT_MAX_BYTE",
        "URLLC_ARRIVAL_DIST","URLLC_ARRIVAL_MEAN_MS","URLLC_ARRIVAL_MAX_MS",
        "URLLC_PKT_DIST","URLLC_PKT_MIN_BYTE","URLLC_PKT_MAX_BYTE",
        "EMBB_QOE_TH","URLLC_QOE_TH","SE_TARGET","SEED",
        "EMBB_HOLD_SLOTS","URLLC_HOLD_SLOTS","LEASE_EPS"
    ]
    with open(snap_path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}={getattr(config, k, None)}\n")
    print(f"[NS][EVAL] wrote snapshot: {snap_path}")

def eval_full(model_path: str, trace_file: str, output_root: str, deterministic: bool = True):
    df = _load_trace_df(trace_file)
    need = {"slot_idx", "embb_bytes", "urllc_bytes"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"[EVAL] Trace must include columns {need}, got {set(df.columns)}")
    total = len(df)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = _ensure_dir(os.path.join(output_root, f"{timestamp}-eval", "output", "eval_full"))

    print(f"[NS][EVAL] Loading model: {model_path}")
    model = DQN.load(model_path)

    dm_backup = getattr(config, "DEMAND_MODE", "TRACE")
    ep_backup = getattr(config, "EP_LEN", 2000)
    trace_backup = getattr(config, "TRACE_FILE", "")

    try:
        config.DEMAND_MODE = "TRACE"
        config.TRACE_FILE  = trace_file
        config.EP_LEN      = int(total)
        config.SLOT_MS     = float(getattr(config, "SLOT_MS", 0.5))
        _snap_config(out_dir)

        env = DummyVecEnv([lambda: NS_Env(seed=getattr(config, "SEED", 42))])
        model.set_env(env)
        obs = env.reset()

        # --- open slots csv and write header (extended with our extra fields) ---
        slots_csv = os.path.join(out_dir, "slots_full.csv")
        header = [
            "t","action","reward",
            "qoe_e","qoe_u",
            "se","se_mbps",
            "utility_se","utility_ese",
            "used_embb_mhz","used_urllc_mhz","used_bw_mhz",
            "utilization","benign_utilization",
            "bandwidth_efficiency","effective_spectrum_efficiency",
        ]



        # final header = base header + extras
        full_header = header

        with open(slots_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(full_header)

        done = False; t = 0; ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            inf = info[0] if isinstance(info,(list,tuple)) else info

            # base row values (keep same order as 'header' above)
            row = [
                t,
                int(action if hasattr(action,"__int__") else action[0]),
                float(reward),
                inf.get("qoe_e", 0.0), inf.get("qoe_u", 0.0),
                inf.get("se", 0.0), inf.get("se_mbps", 0.0),
                inf.get("utility_se", 0.0), inf.get("utility_ese", 0.0),
                inf.get("used_embb_mhz", 0.0), inf.get("used_urllc_mhz", 0.0), inf.get("used_bw_mhz", 0.0),
                inf.get("utilization", 0.0), inf.get("benign_utilization", 0.0),
                inf.get("bandwidth_efficiency", 0.0), inf.get("effective_spectrum_efficiency", 0.0),
            ]


            # write the row
            with open(slots_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            t += 1


        with open(os.path.join(out_dir, "summary_full.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["rows","episode_return"])
            csv.writer(f).writerow([t, f"{ep_return:.6f}"])

        print(f"[NS][EVAL] Done: rows={t}, episode_return={ep_return:.3f}")
        print(f"[NS][EVAL] Output: {slots_csv}")
        return ep_return
    finally:
        config.DEMAND_MODE = dm_backup
        config.EP_LEN = ep_backup
        config.TRACE_FILE = trace_backup

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .zip model (DQN)")
    parser.add_argument("--trace", type=str, default=None, help="Path to trace csv")
    parser.add_argument("--det", action="store_true", help="Deterministic policy for eval")
    parser.add_argument("--out_root", type=str, default=RUNS_ROOT, help="Output root folder")
    args = parser.parse_args()

    model_path = args.model or getattr(config, "MODEL_PATH", None)
    trace_path = args.trace or getattr(config, "TRACE_FILE", None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"[EVAL] model not found: {model_path}")
    if not trace_path or not os.path.exists(trace_path):
        raise FileNotFoundError(f"[EVAL] trace not found: {trace_path}")

    deterministic = True
    print(f"[NS][EVAL] det={deterministic}")
    eval_full(model_path=model_path, trace_file=trace_path, output_root=args.out_root, deterministic=deterministic)

if __name__ == "__main__":
    main()
