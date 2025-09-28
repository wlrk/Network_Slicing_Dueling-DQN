# build_trace_from_detector.py
# Pipeline: (1) drop DDoS (pred==1) → (2) de-normalize (frame.len×1500, ip.proto→6/17)
# → (3) slot at 0.5ms → (4) aggregate eMBB/URLLC (TRACE) + (5) per-record label CSV
import os, csv
import numpy as np
import pandas as pd
from typing import Tuple

# ====== default paths ======
ROOT = os.path.dirname(os.path.abspath(__file__))
DET  = os.path.join(ROOT, "hdf5_predictions.csv")
OUT_TRACE   = os.path.join(ROOT, "runs", "trace_from_detector", "trace_ep2000_slot0p5ms.csv")
OUT_LABELS  = os.path.join(ROOT, "runs", "trace_from_detector", "trace_with_labels.csv")

# ====== params ======
SLOT_MS = 0.5                 # slot width (ms)
LABEL_COL = "pred"            # DDoS flag (1=attack, 0=benign)
EP_LEN = None                 # if None, use full trace length (max slot + 1)
ASSIGN_MODE = "size_cut"      # {"size_cut","none","heuristic"}; trace split rule
SIZE_CUT_BYTE = 600           # size threshold (bytes) for split

# de-normalization
ASSUMED_MAX_BYTE = 1500       # scale to restore frame.len (MTU≈1500)
PROTO_THRESH_HIGH = 0.9       # normalized ip.proto ≥ 0.9 → UDP(17)
PROTO_THRESH_TCP  = 0.3       # normalized ip.proto ≥ 0.3 → TCP(6)

PROB_THRESHOLD = 0.5          # threshold if pred is probability

# record labeling (stricter mode) — UDP & small pkt → URLLC, else eMBB
LABEL_SIZE_THRESH = 600       # should match SIZE_CUT_BYTE for consistency

# include/exclude attack traffic in aggregated trace
INCLUDE_DDOS_IN_TRACE = 1  # 1=include attack bytes in trace, 0=exclude

# ----------------------------------------
# utils
# ----------------------------------------
def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-16", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def _binarize_pred_column(df: pd.DataFrame, col: str, pth: float) -> None:
    if col not in df.columns:
        raise ValueError(f"missing label column: {col}")
    lab = df[col]
    if np.issubdtype(lab.dtype, np.floating) or np.issubdtype(lab.dtype, np.integer):
        if lab.max() <= 1.0 and lab.min() >= 0.0 and lab.max() > 0.01:
            print(f"[repair] {col} looks like probability → threshold {pth}")
            df[col] = (lab.astype(float) >= float(pth)).astype(int)
        else:
            df[col] = lab.astype(int)
    else:
        # text labels: "normal/benign/0" → 0, otherwise → 1
        df[col] = (~lab.astype(str).str.lower().isin(["0","normal","benign"])).astype(int)

def _denorm_frame_len(df: pd.DataFrame) -> None:
    # restore frame length in bytes
    if "frame.len" in df.columns:
        s = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0)
        if s.max() <= 1.01:
            df["frame.len"] = (s * ASSUMED_MAX_BYTE).round().astype(int)
        else:
            df["frame.len"] = s.round().astype(int)
    elif "frame_length" in df.columns:
        s = pd.to_numeric(df["frame_length"], errors="coerce").fillna(0)
        if s.max() <= 1.01:
            df["frame.len"] = (s * ASSUMED_MAX_BYTE).round().astype(int)
        else:
            df["frame.len"] = s.round().astype(int)
    else:
        df["frame.len"] = 0

def _map_ip_proto(df: pd.DataFrame) -> None:
    # restore ip.proto from normalized/unknown to {6,17} when possible
    if "ip.proto" not in df.columns:
        df["ip.proto"] = 0
        return
    s = pd.to_numeric(df["ip.proto"], errors="coerce").fillna(0.0)
    vmax = s.max()
    if vmax <= 1.01:
        proto = np.zeros(len(s), dtype=int)
        proto[s >= PROTO_THRESH_HIGH] = 17
        proto[(s >= PROTO_THRESH_TCP) & (s < PROTO_THRESH_HIGH)] = 6
        df["ip.proto"] = proto
    else:
        df["ip.proto"] = s.astype(int)

def pick_ts_seconds(df: pd.DataFrame) -> np.ndarray:
    # pick timestamp column (seconds)
    for c in ("frame.time_epoch","timestamp","ts","time","time_epoch"):
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            return t
    # fallback: *_ms columns → seconds
    for c in ("ts_ms","timestamp_ms","time_ms"):
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy() / 1000.0
            return t
    # last resort: sequential slots
    n = len(df)
    return np.arange(n, dtype=float) * (SLOT_MS / 1000.0)

def classify_slice_strict(proto: int, plen: int) -> str:
    # URLLC only when UDP and small; otherwise eMBB
    if proto == 17 and plen <= LABEL_SIZE_THRESH:
        return "URLLC"
    return "EMBB"

def assign_split(mode: str, part: pd.DataFrame, size_cut: int) -> Tuple[int, int]:
    """
    Aggregate (embb_bytes, urllc_bytes) within a slot.
      - size_cut: split by packet size
      - heuristic: UDP(17) & frame.len<=256 → URLLC
      - none: all to eMBB
    """
    part_bytes = pd.to_numeric(part.get("__bytes", 0), errors="coerce").fillna(0).astype(int)
    if mode == "none":
        return int(part_bytes.sum()), 0
    if mode == "size_cut":
        url_mask = (part_bytes <= size_cut)
        u = int(part_bytes[url_mask].sum())
        e = int(part_bytes[~url_mask].sum())
        return e, u
    # heuristic
    ip_proto = pd.to_numeric(part.get("ip.proto", 0), errors="coerce").fillna(0).astype(int)
    frame_len = pd.to_numeric(part.get("frame.len", 0), errors="coerce").fillna(0).astype(int)
    url_mask = (ip_proto == 17) & (frame_len <= 256)
    u = int(part.loc[url_mask, "__bytes"].sum())
    e = int(part.loc[~url_mask, "__bytes"].sum())
    return e, u

# ----------------------------------------
# main
# ----------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_TRACE), exist_ok=True)

    # 0) load & restore columns
    df = _read_csv(DET)
    _binarize_pred_column(df, LABEL_COL, PROB_THRESHOLD)
    _denorm_frame_len(df)
    _map_ip_proto(df)

    # 1) include/exclude attacks for trace aggregation
    if LABEL_COL not in df.columns:
        raise ValueError(f"missing label column: {LABEL_COL}")
    lab = df[LABEL_COL]
    is_attack = (lab.astype(str).str.lower().isin(["1", "attack"])) if lab.dtype == object else (lab != 0)
    if int(INCLUDE_DDOS_IN_TRACE) == 0:
        df = df.loc[~is_attack].copy()    # exclude attacks
    else:
        df = df.copy()                    # include attacks; split later

    # preserve original index (if present), else use positional index
    if "index" in df.columns:
        idx_col = df["index"].astype(int).values
    else:
        idx_col = np.arange(len(df), dtype=int)

    if len(df) == 0:
        # write empty outputs
        with open(OUT_TRACE, "w", newline="") as f:
            csv.writer(f).writerow(["slot_idx", "embb_bytes", "urllc_bytes"])
        with open(OUT_LABELS, "w", newline="") as f:
            csv.writer(f).writerow(["index","slot_idx","ip.proto_denorm","frame.len_denorm","slice_label"])
        print("no normal traffic → empty trace & labels written")
        print("trace :", OUT_TRACE)
        print("labels:", OUT_LABELS)
        return

    # 2) slot index
    t_sec = pick_ts_seconds(df)
    slot_idx = np.floor((t_sec * 1000.0) / SLOT_MS).astype(int)
    slot_idx[slot_idx < 0] = 0
    df["__slot"]  = slot_idx
    df["__bytes"] = pd.to_numeric(df.get("frame.len", 0), errors="coerce").fillna(0).astype(int)

    # 3) per-record labels CSV
    ip_den  = pd.to_numeric(df["ip.proto"], errors="coerce").fillna(0).astype(int)
    len_den = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0).astype(int)
    slice_labels = [classify_slice_strict(int(p), int(l)) for p, l in zip(ip_den, len_den)]

    labels_df = pd.DataFrame({
        "slot_idx": df["__slot"].astype(int),
        "index": idx_col,
        "ip.proto_denorm": ip_den.values,
        "frame.len_denorm": len_den.values,
        "slice_label": slice_labels
    })
    labels_df.to_csv(OUT_LABELS, index=False)
    print(f"labels written: {OUT_LABELS}, rows={len(labels_df)}")

    # 4) per-slot aggregation (+ split legit/ddos)
    groups = dict(tuple(df.groupby("__slot", sort=True)))
    rows = []
    target_len = (int(df["__slot"].max()) + 1) if EP_LEN is None else int(EP_LEN)
    for s in range(target_len):
        if s in groups:
            part = groups[s]
            # total split (same as before)
            e_bytes, u_bytes = assign_split(ASSIGN_MODE, part, SIZE_CUT_BYTE)
            # split by legit/ddos
            lab_ser = pd.to_numeric(part.get(LABEL_COL, 0), errors="coerce").fillna(0).astype(int)
            atk_mask = (lab_ser != 0)
            # URLLC mask for the chosen mode
            if ASSIGN_MODE == "size_cut":
                url_mask = (pd.to_numeric(part.get("__bytes", 0), errors="coerce").fillna(0).astype(int) <= SIZE_CUT_BYTE)
            elif ASSIGN_MODE == "heuristic":
                ip_proto = pd.to_numeric(part.get("ip.proto", 0), errors="coerce").fillna(0).astype(int)
                frame_len = pd.to_numeric(part.get("frame.len", 0), errors="coerce").fillna(0).astype(int)
                url_mask = (ip_proto == 17) & (frame_len <= 256)
            else:
                url_mask = (part.index == part.index) & False  # none: no URLLC
            bytes_series = pd.to_numeric(part.get("__bytes", 0), errors="coerce").fillna(0).astype(int)
            req_e_ddos  = int(bytes_series[ atk_mask & (~url_mask) ].sum())
            req_u_ddos  = int(bytes_series[ atk_mask & ( url_mask) ].sum())
            req_e_legit = int(bytes_series[ (~atk_mask) & (~url_mask) ].sum())
            req_u_legit = int(bytes_series[ (~atk_mask) & ( url_mask) ].sum())
        else:
            e_bytes, u_bytes = 0, 0
            req_e_ddos = req_u_ddos = req_e_legit = req_u_legit = 0

        if int(INCLUDE_DDOS_IN_TRACE) == 1:
            embb_bytes = req_e_legit + req_e_ddos
            urllc_bytes = req_u_legit + req_u_ddos
        else:
            embb_bytes = req_e_legit
            urllc_bytes = req_u_legit
            # make ddos demand explicitly zero
            req_e_ddos = 0
            req_u_ddos = 0

        rows.append((s, int(embb_bytes), int(urllc_bytes),
                     int(req_e_legit), int(req_e_ddos), int(req_u_legit), int(req_u_ddos)))

    with open(OUT_TRACE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot_idx","embb_bytes","urllc_bytes","req_e_legit","req_e_ddos","req_u_legit","req_u_ddos"])
        w.writerows(rows)

    emb_sum = sum(r[1] for r in rows)
    url_sum = sum(r[2] for r in rows)
    print(f"trace written: {OUT_TRACE}, rows={len(rows)}, embb_total_bytes={emb_sum}, urllc_total_bytes={url_sum}")

    # 5) quick sanity: TCP labeled as URLLC should be zero
    tcp_url = labels_df[(labels_df["ip.proto_denorm"] == 6) & (labels_df["slice_label"] == "URLLC")]
    if len(tcp_url) > 0:
        print(f"[WARN] Found {len(tcp_url)} rows with TCP(6) labeled URLLC (should be 0). Check mapping/thresholds.")

if __name__ == "__main__":
    main()
