# envs/network_env.py — reservation holding + paper reward + utility logging
import gym
import numpy as np
from gym import spaces
from dataclasses import dataclass
import math
import pandas as pd
import config as C

# ---------- helpers ----------
def dbm_to_w(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)

def noise_psd_w_per_hz() -> float:
    n0_dbm_per_hz = float(getattr(C, "NOISE_PSD_DBM_PER_HZ", -174.0))
    return 10 ** ((n0_dbm_per_hz - 30.0) / 10.0)

def required_bw_hz_for_bps(R_bps: float, tx_dbm: float, g_lin: float, W_total_hz: float) -> float:
    """
    Required bandwidth for a given rate R using Shannon capacity:
    R = W * log2(1 + SNR), with SNR based on total W_total.
    """
    if R_bps <= 0.0: return 0.0
    Tx = dbm_to_w(tx_dbm)
    N0 = noise_psd_w_per_hz()
    snr_total = (Tx * g_lin) / max(N0 * W_total_hz, 1e-30)
    se_total = math.log2(1.0 + max(snr_total, 1e-30))
    if se_total <= 1e-12:
        return W_total_hz
    need = R_bps / se_total
    return min(max(need, 0.0), W_total_hz)

@dataclass
class BWPair:
    embb_mhz: float
    urllc_mhz: float

# ---------- synthetic helpers ----------
def _pareto_xm_from_mean(mean: float, alpha: float) -> float:
    return max(1e-12, mean * (alpha - 1.0) / alpha)

def _trunc(x, lo, hi): return max(lo, min(hi, x))

def sample_interarrival_ms(rng: np.random.RandomState, svc: str) -> float:
    """
    Sample inter-arrival time (ms) based on service type and config distribution.
    """
    dist = str(getattr(C, f"{svc}_ARRIVAL_DIST", "exp")).lower()
    mean_ms = float(getattr(C, f"{svc}_ARRIVAL_MEAN_MS", 5.0))
    max_ms  = float(getattr(C, f"{svc}_ARRIVAL_MAX_MS", mean_ms * 5.0))
    if dist == "const":   return _trunc(mean_ms, 0.0, max_ms)
    if dist == "uniform": return _trunc(float(rng.uniform(0.0, max_ms)), 0.0, max_ms)
    if dist == "pareto_trunc":
        alpha = float(getattr(C, f"{svc}_PARETO_ALPHA", 1.5))
        xm = _pareto_xm_from_mean(mean_ms, alpha)
        v  = xm * (1.0 + rng.pareto(alpha))
        return _trunc(float(v), 0.0, max_ms)
    return _trunc(float(rng.exponential(scale=mean_ms)), 0.0, max_ms)

def sample_packet_bytes(rng: np.random.RandomState, svc: str) -> int:
    """
    Sample packet size (bytes) for given service.
    """
    dist   = str(getattr(C, f"{svc}_PKT_DIST", "const")).lower()
    constb = float(getattr(C, f"{svc}_PKT_CONST_BYTE", getattr(C, f"{svc}_PKT_MEAN_BYTE", 200.0)))
    meanb  = float(getattr(C, f"{svc}_PKT_MEAN_BYTE", constb))
    minb   = float(getattr(C, f"{svc}_PKT_MIN_BYTE", max(1.0, 0.5 * meanb)))
    maxb   = float(getattr(C, f"{svc}_PKT_MAX_BYTE", 4.0 * meanb))
    if dist == "const":           return int(round(_trunc(constb, minb, maxb)))
    if dist == "uniform_trunc":   return int(round(_trunc(float(rng.uniform(minb, maxb)), minb, maxb)))
    if dist == "lognormal_trunc":
        mu  = math.log(max(meanb, 1.0)) - 0.5
        sig = 0.5
        v = float(np.random.lognormal(mean=mu, sigma=sig))
        return int(round(_trunc(v, minb, maxb)))
    return int(round(_trunc(meanb, minb, maxb)))

class NetworkSlicingEnv(gym.Env):
    """
    Environment for eMBB/URLLC slicing with per-request reservation holding.
    Supports both synthetic arrivals and trace-driven arrivals.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.rng = np.random.RandomState(seed=seed)

        # time setup
        self.EP_LEN  = int(getattr(C, "EP_LEN", 2000) or 2000)
        self.slot_ms = float(getattr(C, "SLOT_MS", 0.5))
        self.slot_sec = self.slot_ms / 1000.0

        # action space: reservation caps (eMBB, URLLC)
        W = float(C.TOTAL_BW_MHZ); res = float(C.BW_RES_MHZ)
        k = int(round(W / res)); W = k * res
        pairs = []
        for i in range(k + 1):
            be = i * res
            for j in range(k + 1 - i):
                bl = j * res
                if be + bl <= W + 1e-9:
                    pairs.append(BWPair(round(be, 10), round(bl, 10)))
        pairs.sort(key=lambda p: (-(p.embb_mhz + p.urllc_mhz), p.embb_mhz))
        self.bw_pairs = pairs
        self.action_space = spaces.Discrete(len(self.bw_pairs))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # queues (age, sizeB)
        self.qe, self.qu = [], []

        # demand source
        self.use_trace = str(getattr(C, "DEMAND_MODE", "SYNTH")).upper() == "TRACE"
        if self.use_trace:
            df = None
            last_err = None
            for enc in ("utf-8","utf-16","cp949"):
                try:
                    df = pd.read_csv(getattr(C, "TRACE_FILE", ""), encoding=enc)
                    break
                except Exception as e:
                    last_err = e
            if df is None:
                raise RuntimeError(f"[TRACE] failed to read TRACE_FILE={getattr(C,'TRACE_FILE',None)}; last_err={last_err}")
            need = {"slot_idx","embb_bytes","urllc_bytes"}
            if not need.issubset(df.columns):
                raise RuntimeError(f"[TRACE] columns must include {need}, got {set(df.columns)}")
            df = df.sort_values("slot_idx").reset_index(drop=True)
            self.trace_e = df["embb_bytes"].astype(int).tolist()
            self.trace_u = df["urllc_bytes"].astype(int).tolist()
            self.trace_len = len(self.trace_e)

            self._has_req_cols = all(c in df.columns for c in
                ["req_e_legit","req_e_ddos","req_u_legit","req_u_ddos"])

            if self._has_req_cols:
                self.req_e_legit = df["req_e_legit"].astype(int).tolist()
                self.req_e_ddos  = df["req_e_ddos"].astype(int).tolist()
                self.req_u_legit = df["req_u_legit"].astype(int).tolist()
                self.req_u_ddos  = df["req_u_ddos"].astype(int).tolist()
            else:
                # normal-only trace: fill zeros
                self.req_e_legit = self.req_e_ddos = [0]*self.trace_len
                self.req_u_legit = self.req_u_ddos = [0]*self.trace_len

            # keep last slot request info
            self._last_req_e_legit = self._last_req_e_ddos = 0
            self._last_req_u_legit = self._last_req_u_ddos = 0

        # synthetic timers
        self._next_e_gap_ms = None
        self._next_u_gap_ms = None

        # reservation state
        self.resv_e = []  # list of (bw_mhz, slots_left)
        self.resv_u = []
        self.used_e_mhz = 0.0
        self.used_u_mhz = 0.0

    # arrivals
    def _enqueue_trace_slot(self):
        if self.trace_len == 0:
            e_bytes = u_bytes = 0
            re_legit = re_ddos = ru_legit = ru_ddos = 0
        else:
            i = self.trace_pos
            e_bytes = int(self.trace_e[i]) if i < self.trace_len else 0
            u_bytes = int(self.trace_u[i]) if i < self.trace_len else 0
            re_legit = int(self.req_e_legit[i]) if i < self.trace_len else 0
            re_ddos  = int(self.req_e_ddos[i])  if i < self.trace_len else 0
            ru_legit = int(self.req_u_legit[i]) if i < self.trace_len else 0
            ru_ddos  = int(self.req_u_ddos[i])  if i < self.trace_len else 0

        self.trace_pos += 1
        if e_bytes > 0: self.qe.append((0.0, e_bytes))
        if u_bytes > 0: self.qu.append((0.0, u_bytes))

        # Store last-slot request stats for info
        self._last_req_e_legit, self._last_req_e_ddos = re_legit, re_ddos
        self._last_req_u_legit, self._last_req_u_ddos = ru_legit, ru_ddos
        return e_bytes, u_bytes


    def _pop_count(self, next_gap_get, next_gap_set, sampler, step_ms) -> int:
        # Number of arrivals within this step
        v = next_gap_get()
        cnt = 0
        if v is None:
            v = sampler(self.rng)
        while v <= step_ms:
            cnt += 1
            v += sampler(self.rng)
        next_gap_set(v - step_ms)
        return cnt

    def _sample_arrivals_synth(self):
        step_ms = self.slot_ms
        if self._next_e_gap_ms is None:
            self._next_e_gap_ms = sample_interarrival_ms(self.rng, "EMBB")
        if self._next_u_gap_ms is None:
            self._next_u_gap_ms = sample_interarrival_ms(self.rng, "URLLC")
        e = self._pop_count(lambda: self._next_e_gap_ms, lambda v: setattr(self, "_next_e_gap_ms", v),
                            lambda: sample_interarrival_ms(self.rng, "EMBB"), step_ms)
        l = self._pop_count(lambda: self._next_u_gap_ms, lambda v: setattr(self, "_next_u_gap_ms", v),
                            lambda: sample_interarrival_ms(self.rng, "URLLC"), step_ms)
        e_bytes = sum(sample_packet_bytes(self.rng, "EMBB") for _ in range(int(e)))
        u_bytes = sum(sample_packet_bytes(self.rng, "URLLC") for _ in range(int(l)))
        if e_bytes > 0: self.qe.append((0.0, e_bytes))
        if u_bytes > 0: self.qu.append((0.0, u_bytes))
        return e_bytes, u_bytes

    def _reward_paper(self, qe: float, ql: float, Y_bps_per_hz: float) -> float:
        # reward (QoE-first + SE target)
        q_th_e = float(getattr(C, "EMBB_QOE_TH", 0.98))
        q_th_u = float(getattr(C, "URLLC_QOE_TH", 0.98))
        W_mhz  = float(getattr(C, "TOTAL_BW_MHZ", 20.0))
        if ql < q_th_u:
            return -3.0 + (ql - 1.0) * 10.0
        elif qe < q_th_e:
            return (ql - 1.0) * 10.0
        else:
            target = float(getattr(C, "SE_TARGET", 380.0)) / W_mhz
            return (Y_bps_per_hz * (0.01 * W_mhz)) if Y_bps_per_hz < target else (5.0 + (Y_bps_per_hz - target) * (0.1 * W_mhz))

    def _action_to_bw(self, a: int):
        p = self.bw_pairs[int(a)]
        return p.embb_mhz, p.urllc_mhz

    def _draw_gain(self):
        # distance loss ~ r^-alpha, Rayleigh fading ~ exp(1)
        alpha = float(getattr(C, "PATHLOSS_ALPHA", 3.5))
        rmin  = float(getattr(C, "CELL_R_MIN_M", 10.0))
        rmax  = float(getattr(C, "CELL_R_MAX_M", 300.0))
        r = float(self.rng.uniform(rmin, rmax))
        pl = (rmin / r) ** alpha
        ray = float(np.random.exponential(scale=1.0))
        return max(1e-9, pl * ray)

    def reset(self):
        self.t = 0
        self.qe, self.qu = [], []
        self.resv_e, self.resv_u = [], []
        self.used_e_mhz = 0.0
        self.used_u_mhz = 0.0
        self.trace_pos = 0
        self._next_e_gap_ms = None
        self._next_u_gap_ms = None
        return np.array([0.0, 0.0], dtype=np.float32)

    def _norm_slot_success(self, e_pkts_ok: int, u_pkts_ok: int):
        # simple 2-dim obs: per-slice success (0..1), could be replaced with paper's obs
        cap_e = int(getattr(C, "OBSLOT_E_MAXPKT", 32))
        cap_u = int(getattr(C, "OBSLOT_U_MAXPKT", 32))
        return np.array([min(1.0, e_pkts_ok / max(1,cap_e)),
                         min(1.0, u_pkts_ok / max(1,cap_u))], dtype=np.float32)

    def step(self, action: int):
        self.t += 1
        W_mhz = float(getattr(C, "TOTAL_BW_MHZ", 20.0))

        # arrivals
        if self.use_trace: e_bytes, u_bytes = self._enqueue_trace_slot()
        else:              e_bytes, u_bytes = self._sample_arrivals_synth()

        # 1) required throughput (bps) from queues this slot ---
        slot_s = self.slot_sec
        req_e_bps = int(sum(sz for _,sz in self.qe) * 8.0 / max(slot_s, 1e-9))
        req_u_bps = int(sum(sz for _,sz in self.qu) * 8.0 / max(slot_s, 1e-9))

        # 2) convert to required bandwidth (MHz) via Shannon inverse ---
        W_hz = W_mhz * 1e6
        g_e = self._draw_gain()
        g_u = self._draw_gain()
        need_e_mhz = required_bw_hz_for_bps(req_e_bps, float(getattr(C,"TX_POWER_EMBB_DBM",23.0)), g_e, W_hz)/1e6 if req_e_bps>0 else 0.0
        need_u_mhz = required_bw_hz_for_bps(req_u_bps, float(getattr(C,"TX_POWER_URLLC_DBM",23.0)), g_u, W_hz)/1e6 if req_u_bps>0 else 0.0
        res = float(getattr(C, "BW_RES_MHZ", 0.1))

        def _snap_up(x: float) -> float:
            if x <= 0.0:
                return 0.0
            return math.ceil(x / res) * res

        need_e_mhz = _snap_up(need_e_mhz)
        need_u_mhz = _snap_up(need_u_mhz)
        
        hold_e = int(round(float(getattr(C,"EMBB_HOLD_SLOTS",20))))
        hold_u = int(round(float(getattr(C,"URLLC_HOLD_SLOTS",4))))

        # 3) action as cap for new reservations (URLLC first) ---
        allow_e, allow_u = self._action_to_bw(action)
        new_u = new_e = 0.0
        free_bw = max(0.0, W_mhz - (self.used_e_mhz + self.used_u_mhz))
        if free_bw > 1e-9 and need_u_mhz > 0:
            new_u = min(need_u_mhz, allow_u, free_bw)
            free_bw -= new_u
        if free_bw > 1e-9 and need_e_mhz > 0:
            min_e = float(getattr(C, "EMBB_MIN_BW_MHZ", 0.0))
            new_e = min(need_e_mhz, allow_e, free_bw)
            if min_e > 0 and self.used_e_mhz + new_e < min_e:
                add = min(min_e - self.used_e_mhz, free_bw, allow_e)
                new_e = max(new_e, add)
            free_bw -= new_e
        if new_u > 1e-9: self.resv_u.append((new_u, max(1, hold_u)))
        if new_e > 1e-9: self.resv_e.append((new_e, max(1, hold_e)))
        self.used_e_mhz = float(sum(bw for bw,_ in self.resv_e))
        self.used_u_mhz = float(sum(bw for bw,_ in self.resv_u))

        # URLLC preemption: reclaim eMBB above minimum guarantee
        if getattr(C, "SLA_URLLC_CLEAR_LOAD", False) and need_u_mhz > 0:
            min_e = float(getattr(C, "EMBB_MIN_BW_MHZ", 0.0))
            can_take = max(0.0, (self.used_e_mhz - min_e))
            take = min(can_take, need_u_mhz - self.used_u_mhz)
            if take > 1e-9:
                # Reduce eMBB reservations (largest-first) and add to URLLC
                taken = 0.0
                new_list = []
                for bw, ttl in sorted(self.resv_e, key=lambda x: -x[0]):
                    if taken >= take - 1e-9:
                        new_list.append((bw, ttl))
                        continue
                    delta = min(bw, take - taken)
                    left = bw - delta
                    taken += delta
                    if left > 1e-9:
                        new_list.append((left, ttl))
                self.resv_e = new_list
                self.used_e_mhz = float(sum(bw for bw,_ in self.resv_e))
                self.resv_u.append((take, max(1, hold_u)))
                self.used_u_mhz += take

        # 4) Reservations aging (decrement TTL, drop expired)
        def _decay(lst):
            new=[]
            for bw,ttl in lst:
                ttl2 = ttl-1
                if ttl2 > 0: new.append((bw, ttl2))
            return new
        self.resv_e = _decay(self.resv_e)
        self.resv_u = _decay(self.resv_u)

        # 5) Throughput from SE
        N0 = noise_psd_w_per_hz()
        Tx_e = dbm_to_w(float(getattr(C,"TX_POWER_EMBB_DBM",23.0)))
        Tx_u = dbm_to_w(float(getattr(C,"TX_POWER_URLLC_DBM",23.0)))
        W_hz = W_mhz * 1e6
        # total SNR over total band
        snr_e = (Tx_e * g_e) / max(N0 * W_hz, 1e-30)
        snr_u = (Tx_u * g_u) / max(N0 * W_hz, 1e-30)
        se_bps_per_hz = math.log2(1.0 + max((snr_e + snr_u) * 0.5, 1e-30))  # simple combine
        re_bps = se_bps_per_hz * (self.used_e_mhz * 1e6)
        rl_bps = se_bps_per_hz * (self.used_u_mhz * 1e6)

        # 5) Serve bytes within this slot
        budget_e = int(re_bps * slot_s / 8.0)
        budget_u = int(rl_bps * slot_s / 8.0)
        e_succ_pkts = u_succ_pkts = 0
        new_qu=[]
        for age, sz in self.qu:
            if budget_u >= sz: budget_u -= sz; u_succ_pkts += 1
            else: new_qu.append((age, sz))
        self.qu = new_qu
        new_qe=[]
        for age, sz in self.qe:
            if budget_e >= sz: budget_e -= sz; e_succ_pkts += 1
            else: new_qe.append((age, sz))
        self.qe = new_qe
        served_e_bytes = int(re_bps * slot_s / 8.0) - budget_e
        served_u_bytes = int(rl_bps * slot_s / 8.0) - budget_u

        # 6) QoE per-slot
        qe = 1.0 if (e_bytes <= 0) else float(min(served_e_bytes, e_bytes) / max(e_bytes, 1))
        ql = 1.0 if (u_bytes <= 0) else float(min(served_u_bytes, u_bytes) / max(u_bytes, 1))

        # 7) SE (effective over total band)
        Y_bps_per_hz = (re_bps + rl_bps) / max(W_hz, 1e-9)  
        se_mbps = (re_bps + rl_bps) / 1e6
        reward = self._reward_paper(qe, ql, Y_bps_per_hz)   

        # === Bandwidth efficiency metrics (per-slot) ===
        W_mhz = float(getattr(C, "TOTAL_BW_MHZ", 20.0))
        W_hz  = W_mhz * 1e6
        used_bw_mhz = self.used_e_mhz + self.used_u_mhz


        # A) Bandwidth efficiency (BE)
        W_mhz = float(getattr(C, "TOTAL_BW_MHZ", 20.0))
        W_hz  = W_mhz * 1e6
        used_bw_mhz = self.used_e_mhz + self.used_u_mhz

        if self._has_req_cols:
            e_tot = self._last_req_e_legit + self._last_req_e_ddos
            u_tot = self._last_req_u_legit + self._last_req_u_ddos
            e_share_benign = (self._last_req_e_legit / e_tot) if e_tot>0 else (1.0 if (self.used_e_mhz>0 or e_bytes>0) else 0.0)
            u_share_benign = (self._last_req_u_legit / u_tot) if u_tot>0 else (1.0 if (self.used_u_mhz>0 or u_bytes>0) else 0.0)
        else:
            # normal-only trace: treat all as benign
            e_share_benign = 1.0 if (self.used_e_mhz>0 or e_bytes>0) else 0.0
            u_share_benign = 1.0 if (self.used_u_mhz>0 or u_bytes>0) else 0.0

        benign_bw_mhz = self.used_e_mhz*e_share_benign + self.used_u_mhz*u_share_benign
        bandwidth_efficiency = (benign_bw_mhz / (used_bw_mhz + 1e-9)) if used_bw_mhz>0 else 0.0
        utilization         = (used_bw_mhz / W_mhz) if W_mhz>0 else 0.0
        benign_utilization  = (benign_bw_mhz / W_mhz) if W_mhz>0 else 0.0

        # === B) SE/ESE ===
        se_eff_bps_per_hz = Y_bps_per_hz                         # 전체대역 기준 SE
        effective_spectrum_efficiency = se_eff_bps_per_hz * bandwidth_efficiency  # ESE

        # === C) utilities ===
        alpha = float(getattr(C, "ALPHA", 0.01))
        beta  = float(getattr(C, "BETA", 1.0))
        eta   = float(getattr(C, "ETA", 3.0))
        utility_se  = alpha * se_eff_bps_per_hz          + beta*qe + eta*ql
        utility_ese = alpha * effective_spectrum_efficiency + beta*qe + eta*ql

        obs = self._norm_slot_success(e_succ_pkts, u_succ_pkts)
        done = (self.t >= self.EP_LEN)

        # === D) info  ===
        info = {
            "qoe_e": qe, "qoe_u": ql,
            "se": se_eff_bps_per_hz,
            "se_mbps": se_mbps,
            "utility_se": utility_se,
            "utility_ese": utility_ese,
            "used_embb_mhz": self.used_e_mhz, "used_urllc_mhz": self.used_u_mhz,
            "used_bw_mhz": used_bw_mhz,
            "utilization": utilization, "benign_utilization": benign_utilization,
            "bandwidth_efficiency": bandwidth_efficiency,
            "effective_spectrum_efficiency": effective_spectrum_efficiency,
        }
        return obs, float(reward), bool(done), info