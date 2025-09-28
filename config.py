# config.py — Dueling DQN for eMBB/URLLC (reservation-hold ready)

# ---------- time/episode ----------
EP_LEN = 2000

# ---------- radio/band ----------
TOTAL_BW_MHZ   = 20.0
BW_RES_MHZ     = 0.1
EMBB_MIN_BW_MHZ = 6.0

# ---------- channel/noise/power ----------
NOISE_PSD_DBM_PER_HZ = -174.0
TX_POWER_EMBB_DBM    = 23.0
TX_POWER_URLLC_DBM   = 23.0

# ---------- geometry/channel ----------
BS_RADIUS_M    = 50.0
PATHLOSS_ALPHA = 3.0

# ---------- eMBB traffic ----------
EMBB_ARRIVAL_DIST    = "pareto_trunc"
EMBB_ARRIVAL_MEAN_MS = 6.0
EMBB_ARRIVAL_MAX_MS  = 25.0
EMBB_PARETO_ALPHA    = 1.3
EMBB_PKT_DIST        = "lognormal_trunc"
EMBB_PKT_MEAN_BYTE   = 1200.0
EMBB_PKT_LOGN_SIGMA  = 0.6
EMBB_PKT_MIN_BYTE    = 300.0
EMBB_PKT_MAX_BYTE    = 1500.0

# ---------- URLLC traffic ----------
URLLC_ARRIVAL_DIST    = "exp"
URLLC_ARRIVAL_MEAN_MS = 4.0
URLLC_ARRIVAL_MAX_MS  = 20.0
URLLC_PKT_DIST        = "uniform_trunc"
URLLC_PKT_MIN_BYTE    = 64.0
URLLC_PKT_MAX_BYTE    = 256.0

# ---------- delay constraints ----------
ENFORCE_DELAY      = True
EMBB_MAX_DELAY_MS  = 10.0
URLLC_MAX_DELAY_MS = 1.0

# ---------- reward thresholds ----------
EMBB_QOE_TH = 0.98
URLLC_QOE_TH = 0.98
SE_TARGET   = 380  # bps/Hz target over total band

# ---------- SLA ----------
EMBB_MIN_RATE_MBPS   = 100.0
URLLC_MIN_RATE_MBPS  = 10.0
SLA_ENFORCE_MIN_RATE = False
SLA_URLLC_CLEAR_LOAD = True
SLA_PRIORITY         = "URLLC_FIRST"

# ---------- utility weights (reporting only) ----------
ALPHA = 1
BETA  = 1
ETA   = 1

# ---------- RL hyperparams ----------
GAMMA                 = 0.99
LEARNING_RATE         = 1e-4
BUFFER_SIZE           = 200000
BATCH_SIZE            = 64
EXPLORATION_FRACTION  = 0.3
EXPLORATION_FINAL_EPS = 0.05
TARGET_NET_UPDATE     = 10_000
LEARNING_STARTS       = 10_000
PRIORITIZED_REPLAY    = True
PR_ALPHA              = 0.6
PR_BETA0              = 0.4

# ---------- QoE smoothing ----------
QOE_MODE      = "CUMULATIVE"  # or "INSTANT"
QOE_EMA_ALPHA = 0.10
QOE_INIT      = 1.0

# ---------- reservation holding ----------
SLOT_MS          = 0.5
EMBB_HOLD_SLOTS  = 200  # ≈100ms @ 0.5ms/slot
URLLC_HOLD_SLOTS = 10   # ≈ 5ms @ 0.5ms/slot
LEASE_EPS        = 1e-6

# ---------- demand source ----------
# DEMAND_MODE = "SYNTH"  # internal generator
# DEMAND_MODE = "STREAM" # detector→stream
DEMAND_MODE = "TRACE"

# Use repo-relative paths for GitHub safety
TRACE_FILE          = r"runs\trace_from_detector\trace_ep2000_slot0p5ms.csv"
MODEL_PATH          = r"runs\20250917-113202_last\output\dueling_dqn_slicing_model.zip"

# ---------- misc ----------
SEED = 42
