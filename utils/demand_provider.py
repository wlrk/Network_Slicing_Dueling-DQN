import csv
import os

class TraceDemandProvider:
    """
    slot_idx,embb_bytes,urllc_bytes CSV를 읽어 매 step마다 (embb_bytes, urllc_bytes)를 반환.
    길이가 ep_len보다 짧으면 0으로 패딩, 길면 컷.
    """
    def __init__(self, trace_csv: str, ep_len: int = None):
        self.path = trace_csv
        self.rows = []
        self.i = 0
        self.ep_len = ep_len
        self._load()

    def _load(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        rows = []
        with open(self.path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    s = int(r.get("slot_idx", "0"))
                    e = int(float(r.get("embb_bytes", "0")))
                    u = int(float(r.get("urllc_bytes", "0")))
                except Exception:
                    continue
                rows.append((s, e, u))
        rows.sort(key=lambda x: x[0])
        if self.ep_len is None:
            # 마지막 슬롯+1까지 사용
            self.ep_len = (rows[-1][0] + 1) if rows else 0
        # 연속 슬롯(0..ep_len-1)로 패딩
        arr = [(0,0)] * self.ep_len
        for s, e, u in rows:
            if 0 <= s < self.ep_len:
                arr[s] = (e, u)
        self.rows = arr

    def reset(self):
        self.i = 0

    def step(self):
        if self.i >= len(self.rows):
            return 0, 0
        e, u = self.rows[self.i]
        self.i += 1
        return int(e), int(u)
