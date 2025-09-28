# utils/slot_metrics.py
def compute_slot_metrics(slot_ms: float, phy_total_mhz: float,
                         srv_e_legit_bytes: int, srv_u_legit_bytes: int,
                         srv_e_ddos_bytes: int,  srv_u_ddos_bytes: int,
                         alloc_e_mhz: float, alloc_u_mhz: float):
    """
    Compute per-slot SE and related metrics.

    Returns dict with:
      se_total_bps_hz, se_legit_phy_bps_hz, se_legit_alloc_bps_hz,
      pct_legit, ddos_bits, alloc_legit_mhz, alloc_ddos_mhz
    """
    slot_s = float(slot_ms) / 1000.0
    bits_legit = 8 * (int(srv_e_legit_bytes or 0) + int(srv_u_legit_bytes or 0))
    bits_ddos  = 8 * (int(srv_e_ddos_bytes or 0) + int(srv_u_ddos_bytes or 0))
    bits_total = bits_legit + bits_ddos

    phy_bw_hz = float(phy_total_mhz) * 1e6
    alloc_legit_hz = float((alloc_e_mhz or 0.0) + (alloc_u_mhz or 0.0)) * 1e6

    def safe_div(a, b):
        try:
            return float(a) / float(b) if float(b) != 0 else 0.0
        except Exception:
            return 0.0

    se_total = safe_div(bits_total, phy_bw_hz * slot_s)
    se_legit_phy = safe_div(bits_legit, phy_bw_hz * slot_s)
    se_legit_alloc = safe_div(bits_legit, alloc_legit_hz * slot_s)

    pct_legit = safe_div(bits_legit, bits_total) if bits_total > 0 else 0.0

    return {
        "se_total_bps_hz": se_total,
        "se_legit_phy_bps_hz": se_legit_phy,
        "se_legit_alloc_bps_hz": se_legit_alloc,
        "pct_legit": pct_legit,
        "ddos_bits": int(bits_ddos),
        "alloc_legit_mhz": float((alloc_e_mhz or 0.0) + (alloc_u_mhz or 0.0)),
        "alloc_ddos_mhz": 0.0,
    }
