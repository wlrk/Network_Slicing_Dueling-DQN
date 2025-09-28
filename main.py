# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import argparse
from contextlib import contextmanager

from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import logger

import config
from envs.network_env import NetworkSlicingEnv


# ---------- Console tee: mirror stdout to file ----------
class TeeStdout:
    def __init__(self, path, mode="w", encoding="utf-8"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.file = open(path, mode, encoding=encoding, newline="")
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self.file.write(s)
    def flush(self):
        self._stdout.flush()
        self.file.flush()
    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
        sys.stdout = self._stdout

@contextmanager
def tee_console(path):
    tee = TeeStdout(path)
    sys.stdout = tee
    try:
        yield
    finally:
        tee.close()


# ---------- Env factory ----------
def make_env(seed=None, output_dir=None):
    def _init():
        env = NetworkSlicingEnv(seed=seed)
        if output_dir is not None:
            log_csv = os.path.join(output_dir, "monitor.csv")
            os.makedirs(output_dir, exist_ok=True)
            # Must match keys provided by env's `info`
            env = Monitor(
                env,
                log_csv,
                allow_early_resets=True,
                info_keywords=('qe_embb', 'ql_urllc', 'se', 'se_mbps', 'utility')
            )
        return env
    return _init


# ---------- (Optional) per-step metrics CSV ----------
def make_metrics_writer(output_dir):
    path = os.path.join(output_dir, "train-metrics.csv")
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timesteps","reward","utility","se","se_mbps",
                "qe_embb","ql_urllc","bw_embb_mhz","bw_urllc_mhz"
            ])

    def write(infos, t, rewards=None):
        if not infos:
            return
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        try:
            rew = float(rewards[0]) if rewards is not None else ""
        except Exception:
            rew = ""
        row = [
            int(t),
            rew,
            info0.get("utility",""),
            info0.get("se",""),
            info0.get("se_mbps",""),
            info0.get("qe_embb",""),
            info0.get("ql_urllc",""),
            info0.get("bw_embb_mhz",""),
            info0.get("bw_urllc_mhz",""),
        ]
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    return write


# ---------- Stable-Baselines callback (episode log + table) ----------
def sb_callback_factory(output_dir):
    """
    - Keep 'episode_reward' visible in SB logs
    - Write per-episode summary to $OUT/episode_log.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "episode_log.csv")
    wrote_header = [False]
    ep_idx = 0
    last_ep_reward = None

    def _cb(locals_, globals_):
        nonlocal ep_idx, last_ep_reward

        # Persist last episode reward in SB table
        if last_ep_reward is not None:
            logger.logkv("episode_reward", float(last_ep_reward))

        infos = locals_.get("infos", None)
        if not infos:
            return True
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        if not isinstance(info0, dict):
            return True

        ep_info = info0.get("episode")
        if ep_info is None:
            return True

        # Just finished episode
        last_ep_reward = float(ep_info.get("r", float("nan")))
        logger.logkv("episode_reward", last_ep_reward)

        row = {
            "episode": ep_idx + 1,
            "r": last_ep_reward,
            "l": ep_info.get("l", ""),
            "t": ep_info.get("t", ""),
            "qe_embb": info0.get("qe_embb", ""),
            "ql_urllc": info0.get("ql_urllc", ""),
            "se": info0.get("se", ""),
            "se_mbps": info0.get("se_mbps", ""),
            "utility": info0.get("utility", ""),
        }
        if not wrote_header[0]:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader(); w.writerow(row)
            wrote_header[0] = True
        else:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)
        
        print(f"[NS][Episode {ep_idx + 1}] reward = {last_ep_reward:.2f}", flush=True)

        ep_idx += 1
        return True
    return _cb


def _auto_output_dir(run_root):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(run_root, stamp, "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--dueling", action="store_true", help="use dueling head")
    parser.add_argument("--run_dir", type=str, default=None, help="runs root; None = config.RUNS_DIR")
    parser.add_argument("--output_dir", type=str, default=None, help="if None, auto-create runs\\YYYYMMDD-HHMMSS\\output")
    args = parser.parse_args()

    # Decide runs root
    run_root = args.run_dir if args.run_dir else getattr(config, "RUNS_DIR", "runs")
    os.makedirs(run_root, exist_ok=True)

    # Auto-create output directory
    if args.output_dir is None:
        args.output_dir = _auto_output_dir(run_root)
    os.makedirs(args.output_dir, exist_ok=True)

    # Execution metadata (reproducibility)
    with open(os.path.join(args.output_dir, "run_metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_dir={os.path.abspath(run_root)}\n")
        f.write(f"output_dir={os.path.abspath(args.output_dir)}\n")
        f.write(f"dueling={args.dueling}\n")

    # SB logger: stdout + csv + tensorboard
    logger.configure(folder=args.output_dir, format_strs=['stdout', 'csv', 'tensorboard'])

    # Vectorized env with Monitor (monitor.csv includes QoE/SE/utility if provided)
    env = DummyVecEnv([make_env(seed=getattr(config, "SEED", 42), output_dir=args.output_dir)])

    # DQN / Dueling DQN config
    policy_kwargs = dict(dueling=bool(args.dueling), layers=[512, 512])
    dqn_kwargs = dict(
        gamma=getattr(config, "GAMMA", 0.99),
        learning_rate=getattr(config, "LEARNING_RATE", 1e-4),
        buffer_size=getattr(config, "BUFFER_SIZE", 200000),
        batch_size=getattr(config, "BATCH_SIZE", 64),
        exploration_fraction=getattr(config, "EXPLORATION_FRACTION", 0.3),
        exploration_final_eps=getattr(config, "EXPLORATION_FINAL_EPS", 0.05),
        target_network_update_freq=getattr(config, "TARGET_NET_UPDATE", 10000),
        learning_starts=getattr(config, "LEARNING_STARTS", 10000),
        prioritized_replay=getattr(config, "PRIORITIZED_REPLAY", True),
        prioritized_replay_alpha=getattr(config, "PR_ALPHA", 0.6),
        prioritized_replay_beta0=getattr(config, "PR_BETA0", 0.4),
        prioritized_replay_beta_iters=getattr(config, "PR_BETA_ITERS", None),  # allow None via getattr
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.output_dir,
        verbose=1
    )

    write_metrics = make_metrics_writer(args.output_dir)
    console_txt = os.path.join(args.output_dir, "train-console.txt")

    with tee_console(console_txt):
        print("[NS] building env & model...", flush=True)
        model = DQN("MlpPolicy", env, **dqn_kwargs)

        print("[NS] starting learn...", flush=True)
        cb = sb_callback_factory(args.output_dir)

        # Wrap: also write per-step metrics
        def _cb(_locals, _globals):
            cb_ok = cb(_locals, _globals)
            infos = _locals.get("infos", None)
            num_timesteps = _locals.get("self").num_timesteps if _locals.get("self") is not None else None
            rewards = _locals.get("rewards", None)
            if infos is not None and (num_timesteps is not None):
                write_metrics(infos, num_timesteps, rewards=rewards)
            return cb_ok

        model.learn(total_timesteps=int(args.timesteps), callback=_cb)

        # Save (name depends on dueling flag)
        model_name = "dueling_dqn_slicing_model.zip" if args.dueling else "dqn_slicing_model.zip"
        save_path = os.path.join(args.output_dir, model_name)
        model.save(save_path)
        print(f"[NS] saved model: {save_path}", flush=True)

        print("[NS] done.", flush=True)


if __name__ == "__main__":
    main()
