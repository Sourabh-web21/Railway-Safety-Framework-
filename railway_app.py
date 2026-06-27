"""
Railway Safety Framework — consolidated runnable script + Gradio UI
====================================================================

This single file reproduces the whole notebook pipeline and adds an interactive
UI to add/remove trains, edit the network, and observe scale-up behaviour.

Sections
--------
  1. CollisionLSTM model
  2. Dataset generation        (cells 4-6, scaled up + reproducible)
  3. Leak-free training        (cells 11-17: temporal split, train-only scaler,
                                val-selected threshold) -> saves model artifacts
  4. RailwaySim engine         (cell 25 "Final" simulation, refactored so the
                                network and trains are configurable, not globals)
  5. Headless metrics          + reroute-vs-no-reroute comparison (the rerouting
                                performance evaluation that was missing)
  6. Dashboard video renderer
  7. Gradio UI

Run
---
  python railway_app.py            # ensure model is trained, then launch the UI
  python railway_app.py --train    # force-retrain the model, then launch the UI
  python railway_app.py --no-ui    # just (re)build artifacts, no UI

Artifacts written next to this file: collision_lstm.pt, scaler.pkl,
model_config.json, train_simulation_dataset.csv
"""

from __future__ import annotations
import os, sys, json, pickle, random, heapq, warnings, argparse
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(HERE, "collision_lstm.pt")
SCALER_PATH = os.path.join(HERE, "scaler.pkl")
CONFIG_PATH = os.path.join(HERE, "model_config.json")
CSV_PATH    = os.path.join(HERE, "train_simulation_dataset.csv")

FEATURES = ["velocity", "acceleration", "estimated_headway", "gradient",
            "curvature", "speed_limit", "signal_state", "friction"]
SEQ_LEN  = 10

# Dataset-generation parameters (documented for reproducibility)
GEN_SEED       = 42
NUM_SEGMENTS   = 12
NUM_TRAINS_GEN = 20
TIMESTEPS      = 6000
EPOCHS         = 100
PATIENCE       = 10
BATCH_SIZE     = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════
# 1. MODEL
# ════════════════════════════════════════════════════════════════════════
class CollisionLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.fc2  = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)


# ════════════════════════════════════════════════════════════════════════
# 2. DATASET GENERATION  (1-D physics simulator, reproducible)
# ════════════════════════════════════════════════════════════════════════
class TrackSegment:
    def __init__(self, segment_id, length=1000):
        self.id = segment_id
        self.length = length
        rng = np.random.RandomState(seed=segment_id * 17 + 3)
        self.gradient     = rng.uniform(-0.02, 0.02)
        self.curvature    = rng.uniform(0.001, 0.012)
        self.speed_limit  = rng.uniform(20.0, 80.0)
        self.friction     = rng.uniform(0.70, 0.95)
        self.signal_state = 0

    def update_signal(self, trains_on_segment, all_trains):
        min_gap = float("inf")
        for t1 in trains_on_segment:
            for t2 in all_trains:
                if t2.id == t1.id:
                    continue
                gap = t2.position - t1.position
                if 0 < gap < min_gap:
                    min_gap = gap
        self.signal_state = 2 if min_gap < 50 else 1 if min_gap < 150 else 0


class GenTrain:
    DRIVER_TYPES = ["aggressive", "normal", "lazy"]

    def __init__(self, train_id, seed=None):
        self.id = train_id
        self.rng = np.random.RandomState(seed)
        self.position     = self.rng.uniform(0, 200)
        self.velocity     = self.rng.uniform(10, 20)
        self.acceleration = 0.0
        self.max_accel    = 1.2
        self.max_brake    = 2.5
        self.driver_type  = self.DRIVER_TYPES[self.rng.randint(3)]
        self.position_history = deque(maxlen=5)

    def update(self, dt, segment):
        if   self.driver_type == "aggressive": accel = self.rng.uniform(0.5, self.max_accel)
        elif self.driver_type == "lazy":       accel = self.rng.uniform(0.1, 0.6)
        else:                                   accel = self.rng.uniform(0.3, 0.9)
        if self.rng.rand() < 0.002:
            accel = -self.max_brake
        if   segment.signal_state == 2 and self.velocity > 5: accel = -self.max_brake * 0.8
        elif segment.signal_state == 1 and self.velocity > 8: accel = -self.max_brake * 0.3
        accel -= segment.gradient * 9.81
        self.acceleration = accel
        self.velocity  = float(np.clip(self.velocity + accel * dt, 0, segment.speed_limit))
        self.position += self.velocity * dt
        self.position_history.append(self.position)


def generate_dataset():
    import pandas as pd
    np.random.seed(GEN_SEED); random.seed(GEN_SEED)
    segments = [TrackSegment(i) for i in range(NUM_SEGMENTS)]
    trains   = [GenTrain(f"T{i}", seed=1000 + i) for i in range(NUM_TRAINS_GEN)]
    records  = []
    print(f"Generating {NUM_TRAINS_GEN} trains x {TIMESTEPS} steps "
          f"= {NUM_TRAINS_GEN*TIMESTEPS:,} rows ...")
    for t in range(TIMESTEPS):
        for seg in segments:
            on_seg = [tr for tr in trains
                      if int(tr.position // 1000) % NUM_SEGMENTS == seg.id]
            seg.update_signal(on_seg, trains)
        for tr in trains:
            seg = segments[int(tr.position // 1000) % NUM_SEGMENTS]
            tr.update(1, seg)
        for tr in trains:
            seg = segments[int(tr.position // 1000) % NUM_SEGMENTS]
            delay = np.random.randint(0, 3)
            hist = list(tr.position_history)
            dpos = hist[-delay - 1] if len(hist) > delay else tr.position
            measured_position = dpos + np.random.normal(0, 3)
            measured_velocity = tr.velocity + np.random.normal(0, 0.5)
            min_headway, closing = 9999.0, 0.0
            for other in trains:
                if other.id == tr.id:
                    continue
                dist = other.position - tr.position
                if 0 < dist < min_headway:
                    min_headway, closing = dist, tr.velocity - other.velocity
            est_headway = float(np.clip(min_headway + np.random.normal(0, 2), 0, None))
            stop_dist = (tr.velocity ** 2) / (2 * tr.max_brake * max(seg.friction, 0.01))
            ttc = min_headway / closing if closing > 0.5 else 9999.0
            label = int(
                (min_headway < stop_dist and ttc < 20 and tr.velocity > 5 and closing > 1.0)
                or (seg.signal_state == 2 and tr.velocity > 10))
            records.append({
                "timestamp": t, "train_id": tr.id, "position": measured_position,
                "velocity": measured_velocity, "acceleration": tr.acceleration,
                "estimated_headway": est_headway, "segment_id": seg.id,
                "gradient": seg.gradient, "curvature": seg.curvature,
                "speed_limit": seg.speed_limit, "signal_state": seg.signal_state,
                "friction": seg.friction, "collision_risk": label})
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset shape: {df.shape}  (saved to {os.path.basename(CSV_PATH)})")
    return df


# ════════════════════════════════════════════════════════════════════════
# 3. LEAK-FREE TRAINING  (temporal split -> train-only scaler -> windows)
# ════════════════════════════════════════════════════════════════════════
def train_pipeline():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_curve
    from torch.utils.data import TensorDataset, DataLoader

    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else generate_dataset()
    df["estimated_headway"] = df["estimated_headway"].clip(upper=500)

    # --- temporal split per train on raw rows (no window/scaler/threshold leak) ---
    tr_rows, va_rows, te_rows = [], [], []
    for tid in df["train_id"].unique():
        g = df[df["train_id"] == tid].sort_values("timestamp")
        n = len(g); a, b = int(n * 0.70), int(n * 0.85)
        tr_rows.append(g.iloc[:a]); va_rows.append(g.iloc[a:b]); te_rows.append(g.iloc[b:])
    train_df, val_df, test_df = pd.concat(tr_rows), pd.concat(va_rows), pd.concat(te_rows)

    def keyset(d): return set(zip(d["train_id"], d["timestamp"]))
    assert keyset(train_df).isdisjoint(keyset(val_df))
    assert keyset(train_df).isdisjoint(keyset(test_df))
    assert keyset(val_df).isdisjoint(keyset(test_df))

    scaler = StandardScaler().fit(train_df[FEATURES])     # train rows only
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    def build(d):
        Xs, ys = [], []
        for tid in d["train_id"].unique():
            g = d[d["train_id"] == tid].sort_values("timestamp")
            vals = scaler.transform(g[FEATURES]); tgt = g["collision_risk"].values
            for i in range(len(g) - SEQ_LEN):
                Xs.append(vals[i:i + SEQ_LEN]); ys.append(tgt[i + SEQ_LEN])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_tr, y_tr = build(train_df); X_va, y_va = build(val_df); X_te, y_te = build(test_df)

    # upsample positives in TRAIN only
    pos, neg = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    if len(pos):
        rng = np.random.RandomState(42)
        up = rng.choice(pos, size=int(len(neg) * 0.10), replace=True)
        idx = np.concatenate([neg, up]); rng.shuffle(idx)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    print(f"train {X_tr.shape} val {X_va.shape} test {X_te.shape} | "
          f"train balance {Counter(y_tr.tolist())}")

    def T(a): return torch.tensor(a, dtype=torch.float32)
    X_tr, y_tr = T(X_tr).to(DEVICE), T(y_tr).unsqueeze(1).to(DEVICE)
    X_va, y_va = T(X_va).to(DEVICE), T(y_va).unsqueeze(1).to(DEVICE)
    X_te, y_te = T(X_te).to(DEVICE), T(y_te).unsqueeze(1).to(DEVICE)

    model = CollisionLSTM(input_size=len(FEATURES)).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=DEVICE))
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    dl    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); run = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = crit(model(X_va), y_va).item()
        print(f"epoch {epoch+1:3d} | train {run/len(X_tr):.4f} | val {vl:.4f}")
        if vl < best_val - 1e-4:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop @ epoch {epoch+1} (best val {best_val:.4f})"); break
    if best_state:
        model.load_state_dict(best_state)

    # threshold selected on VALIDATION, reported on TEST
    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(X_va)).cpu().numpy().flatten()
        tp = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
    p, r, th = precision_recall_curve(y_va.cpu().numpy().flatten(), vp)
    f1 = 2 * p * r / (p + r + 1e-8)
    best_thresh = float(th[f1.argmax()])

    yt = y_te.cpu().numpy().flatten(); pred = (tp > best_thresh).astype(float)
    tp_, fp_ = int(((pred == 1) & (yt == 1)).sum()), int(((pred == 1) & (yt == 0)).sum())
    fn_ = int(((pred == 0) & (yt == 1)).sum())
    prec = tp_ / (tp_ + fp_ + 1e-9); rec = tp_ / (tp_ + fn_ + 1e-9)
    test_f1 = 2 * prec * rec / (prec + rec + 1e-9)

    torch.save(model.state_dict(), MODEL_PATH)
    cfg = {"optimal_threshold": best_thresh, "features": FEATURES,
           "sequence_length": SEQ_LEN, "hidden_size": 64,
           "test_precision": round(prec, 4), "test_recall": round(rec, 4),
           "test_f1": round(test_f1, 4)}
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved model -> {os.path.basename(MODEL_PATH)} | "
          f"thresh={best_thresh:.3f} | TEST P={prec:.3f} R={rec:.3f} F1={test_f1:.3f}")
    return model, scaler, cfg


def load_model():
    """Load model + scaler + config; train if any artifact is missing."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
            and os.path.exists(CONFIG_PATH)):
        return train_pipeline()
    model = CollisionLSTM(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return model, scaler, cfg


# ════════════════════════════════════════════════════════════════════════
# 4. SIMULATION ENGINE  (configurable network + trains)
# ════════════════════════════════════════════════════════════════════════
DEFAULT_NETWORK = {
    "pos": {
        "W": [0, 6], "C": [6, 6], "E": [12, 6], "S": [6, 1],
        "mWC": [3, 6], "mCE": [9, 6], "mWS": [2, 2], "mSE": [10, 2],
        "mCS": [6, 3], "mNE": [10, 8], "mNW": [2, 8],
    },
    "majors": ["W", "C", "E", "S"],
    "edges": [
        ["W", "mWC", 3], ["mWC", "C", 3], ["C", "mCE", 3], ["mCE", "E", 3],
        ["W", "mWS", 4], ["mWS", "S", 4], ["S", "mSE", 4], ["mSE", "E", 4],
        ["C", "mCS", 3], ["mCS", "S", 3],
        ["W", "mNW", 5], ["mNW", "C", 5], ["C", "mNE", 5], ["mNE", "E", 5],
    ],
}

DEFAULT_TRAINS = [
    {"id": "T1", "route": ["W", "C", "E"], "priority": 1, "speed": [0.013, 0.018]},
    {"id": "T2", "route": ["E", "C", "W"], "priority": 2, "speed": [0.011, 0.016]},
    {"id": "T3", "route": ["S", "C", "W"], "priority": 2, "speed": [0.011, 0.016]},
    {"id": "T4", "route": ["W", "S", "E"], "priority": 2, "speed": [0.010, 0.015]},
    {"id": "T5", "route": ["C", "W", "S"], "priority": 3, "speed": [0.009, 0.014]},
    {"id": "T6", "route": ["C", "S", "E"], "priority": 3, "speed": [0.009, 0.014]},
]

# tuning constants (from the original "Final" simulation)
MAX_ON_SEG          = 2
PATH_REROUTE_THRESH = 0.25
CONGESTION_PENALTY  = 15.0
HEAD_ON_DIST        = 1.20
HEAD_ON_FRAMES      = 60
CRITICAL_DIST, WARNING_DIST, SAFE_DIST = 0.35, 0.70, 1.50
SIGNAL_RED_M, SIGNAL_YELLOW_M = 100, 250
HIST_LEN = 40


class SimTrain:
    def __init__(self, spec, path, stagger):
        self.train_id  = spec["id"]
        self.mandatory = list(spec["route"])
        self.priority  = int(spec.get("priority", 3))
        lo, hi = spec.get("speed", [0.010, 0.016])
        self.base_speed = random.uniform(lo, hi)
        self.path       = path
        self.edge_index = 0
        self.progress   = stagger
        self.speed      = self.base_speed
        self.history    = deque(maxlen=50)
        self.is_braking = False
        self.brake_intensity = 0.0
        self.velocity   = 0.0
        self.collision_risk = 0.0
        self.reroute_count  = 0
        self.last_reroute   = -9999
        self.reroute_flash  = 0
        self.stopped_headon = False
        self.headon_timer   = 0
        self.completed_edges = 0

    def current_edge(self):
        if self.edge_index >= len(self.path) - 1:
            return None
        return (self.path[self.edge_index], self.path[self.edge_index + 1])

    def apply_brake(self, intensity):
        self.is_braking = True; self.brake_intensity = intensity
        self.speed = self.base_speed * (1 - intensity * 0.8)

    def release_brake(self):
        self.is_braking = False; self.brake_intensity = 0.0; self.speed = self.base_speed

    def full_stop(self, frames):
        self.stopped_headon = True; self.headon_timer = frames
        self.speed = 0.0; self.brake_intensity = 1.0; self.is_braking = True

    def update(self):
        if self.stopped_headon:
            self.headon_timer -= 1
            if self.headon_timer <= 0:
                self.stopped_headon = False; self.brake_intensity = 0.0
                self.is_braking = False; self.speed = self.base_speed
            return
        self.progress += self.speed
        if self.progress >= 1:
            if self.edge_index < len(self.path) - 1:
                self.completed_edges += 1
                self.edge_index += 1
                self.progress = 0
            else:
                # arrived at the final station: park there, don't wrap around
                self.progress = 1.0
        if self.reroute_flash > 0:
            self.reroute_flash -= 1


class RailwaySim:
    """Holds its own network + trains + model so multiple configs can run."""

    def __init__(self, network, train_specs, model, scaler, threshold,
                 seed=0, enable_reroute=True):
        import networkx as nx
        random.seed(seed); np.random.seed(seed)
        self.model, self.scaler, self.threshold = model, scaler, threshold
        self.enable_reroute = enable_reroute

        self.pos    = {k: tuple(v) for k, v in network["pos"].items()}
        self.majors = set(network["majors"])
        self.minors = set(self.pos) - self.majors
        self.G = nx.Graph()
        self.G.add_weighted_edges_from([(u, v, w) for u, v, w in network["edges"]])

        self._canon = {}
        for u, v in self.G.edges():
            self._canon[(u, v)] = (u, v); self._canon[(v, u)] = (u, v)
        self.SEGMENTS = sorted({self._canon[(u, v)] for u, v in self.G.edges()})

        # per-segment physical properties (deterministic), used by the LSTM
        self.seg_props = {}
        for i, s in enumerate(self.SEGMENTS):
            rng = np.random.RandomState(i * 17 + 3)
            self.seg_props[s] = {
                "gradient": rng.uniform(-0.02, 0.02), "curvature": rng.uniform(0.001, 0.012),
                "speed_limit": rng.uniform(20.0, 80.0), "friction": rng.uniform(0.70, 0.95)}

        self.seg_history = {s: deque([0] * HIST_LEN, maxlen=HIST_LEN) for s in self.SEGMENTS}
        self.seg_load = {s: 0 for s in self.SEGMENTS}
        self.seg_pred = {s: 0.0 for s in self.SEGMENTS}
        self.station_signal = {s: 0 for s in self.majors}

        # build trains
        staggers = [0.10, 0.10, 0.10, 0.40, 0.75, 0.50]
        self.trains = []
        for i, spec in enumerate(train_specs):
            path = self.mandatory_route(spec["route"]) or list(spec["route"])
            self.trains.append(SimTrain(spec, path, staggers[i % len(staggers)]))

        # metrics accumulators
        self.min_headway_hist = []
        self.avg_risk_hist    = []
        self.headon_events    = 0
        self._prev_stopped    = set()

    # ---- graph helpers ----
    def canon(self, a, b): return self._canon[(a, b)]

    def build_cost_graph(self):
        cost = {}
        for u, v, d in self.G.edges(data=True):
            base = d.get("weight", 3); s = self.canon(u, v)
            pen = CONGESTION_PENALTY * self.seg_pred[s] if self.seg_pred[s] > 0.40 else 0
            w = base + pen
            cost.setdefault(u, {})[v] = w; cost.setdefault(v, {})[u] = w
        return cost

    @staticmethod
    def dijkstra(graph, src, dst):
        dist = {src: 0}; prev = {}; pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")): continue
            if u == dst: break
            for v, w in graph.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        path, cur = [], dst
        while cur != src:
            path.append(cur); cur = prev.get(cur)
            if cur is None: return None
        path.append(src); return list(reversed(path))

    def mandatory_route(self, waypoints):
        cg = self.build_cost_graph(); full = []
        for i in range(len(waypoints) - 1):
            seg = self.dijkstra(cg, waypoints[i], waypoints[i + 1])
            if seg is None: return None
            full = full[:-1] + seg if full else seg
        return full

    def score_path(self, p):
        return sum(self.seg_pred.get(self.canon(p[i], p[i + 1]), 0.0) for i in range(len(p) - 1))

    def train_pos(self, t):
        if t.edge_index >= len(t.path) - 1:
            return self.pos[t.path[-1]]
        x1, y1 = self.pos[t.path[t.edge_index]]; x2, y2 = self.pos[t.path[t.edge_index + 1]]
        return (x1 + (x2 - x1) * t.progress, y1 + (y2 - y1) * t.progress)

    def choose_best_path(self, t, src, dst):
        import networkx as nx
        try:
            allp = list(nx.all_simple_paths(self.G, src, dst, cutoff=6))
        except Exception:
            allp = []
        visited = set(t.path[:t.edge_index + 1])
        unvisited = set(t.mandatory) - visited - {src, dst}
        valid = [p for p in allp if not set(p[1:-1]).intersection(unvisited)]
        if not valid:
            return self.dijkstra(self.build_cost_graph(), src, dst)
        return min(valid, key=self.score_path)

    # ---- per-step subsystems ----
    def update_occupancy(self):
        for s in self.SEGMENTS: self.seg_load[s] = 0
        for t in self.trains:
            e = t.current_edge()
            if e: self.seg_load[self.canon(*e)] += 1
        for s in self.SEGMENTS:
            self.seg_history[s].append(self.seg_load[s])
            avg = np.mean(list(self.seg_history[s])[-10:]) / MAX_ON_SEG
            curr = self.seg_load[s] / MAX_ON_SEG
            self.seg_pred[s] = float(np.clip(0.6 * avg + 0.4 * curr, 0, 1))

    def update_signals(self):
        for st in self.majors:
            sx, sy = self.pos[st]; mind = float("inf")
            for t in self.trains:
                tx, ty = self.train_pos(t)
                mind = min(mind, np.linalg.norm([tx - sx, ty - sy]) * 100)
            self.station_signal[st] = 2 if mind < SIGNAL_RED_M else 1 if mind < SIGNAL_YELLOW_M else 0

    def detect_headon(self):
        handled = set()
        for i, t1 in enumerate(self.trains):
            if t1.stopped_headon: continue
            e1 = t1.current_edge()
            if e1 is None: continue
            for t2 in self.trains[i + 1:]:
                if t2.stopped_headon: continue
                e2 = t2.current_edge()
                if e2 is None: continue
                if e2 == (e1[1], e1[0]):
                    d = np.linalg.norm(np.array(self.train_pos(t1)) - np.array(self.train_pos(t2)))
                    if d < HEAD_ON_DIST and (t1.train_id, t2.train_id) not in handled:
                        t1.full_stop(HEAD_ON_FRAMES); t2.full_stop(HEAD_ON_FRAMES)
                        handled.add((t1.train_id, t2.train_id))

    def compute_risk(self, t):
        ce = t.current_edge()
        if ce is None: return 0.0
        min_hw = 10000.0
        for o in self.trains:
            if o.train_id == t.train_id: continue
            oe = o.current_edge()
            if oe and (oe == ce or oe == (ce[1], ce[0])):
                d = np.linalg.norm(np.array(self.train_pos(t)) - np.array(self.train_pos(o)))
                min_hw = min(min_hw, d * 100)
        velocity = t.speed * 100.0
        accel    = (t.speed - t.base_speed) * 100.0
        headway  = float(np.clip(min_hw, 0, 500))
        props    = self.seg_props[self.canon(*ce)]
        sig = 2.0 if min_hw < 100 else 1.0 if min_hw < 250 else 0.0
        raw = np.array([velocity, accel, headway, props["gradient"], props["curvature"],
                        props["speed_limit"], sig, props["friction"]], dtype=np.float32)
        feat = self.scaler.transform(raw.reshape(1, -1))[0].astype(np.float32) \
            if self.scaler is not None else raw
        t.history.append(feat)
        if len(t.history) >= SEQ_LEN:
            seq = np.array(list(t.history)[-SEQ_LEN:])
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(x)).item()
        else:
            prob = 1.0 if min_hw < CRITICAL_DIST * 100 else \
                   0.5 if min_hw < WARNING_DIST * 100 else 0.1
        t.velocity = velocity; t.collision_risk = prob
        if   min_hw < CRITICAL_DIST * 100:
            t.apply_brake(min(1.0, (CRITICAL_DIST * 100 - min_hw) / (CRITICAL_DIST * 100)))
        elif min_hw < WARNING_DIST * 100: t.apply_brake(0.5)
        elif min_hw < SAFE_DIST * 100:    t.apply_brake(0.2)
        elif not t.stopped_headon:        t.release_brake()
        return prob

    def reroute_one(self, frame):
        cands = []
        for t in self.trains:
            if t.stopped_headon or t.priority == 1: continue
            if t.path[t.edge_index] not in self.majors: continue
            if t.progress >= 0.20 or frame - t.last_reroute < 30: continue
            visited = {n for n in t.path[:t.edge_index + 1] if n in self.majors}
            remaining = [m for m in t.mandatory if m not in visited]
            if not remaining: continue
            nm = remaining[0]
            try:
                k = t.path[t.edge_index:].index(nm)
                cur_seg = t.path[t.edge_index: t.edge_index + k + 1]
            except ValueError:
                continue
            if self.score_path(cur_seg) / max(len(cur_seg) - 1, 1) < PATH_REROUTE_THRESH:
                continue
            cands.append((t.priority, id(t), t, remaining, nm, cur_seg))
        if not cands: return
        cands.sort(key=lambda x: -x[0])
        _, _, t, remaining, nm, cur_seg = cands[0]
        best = self.choose_best_path(t, t.path[t.edge_index], nm)
        if best is None: return
        if self.score_path(best) >= self.score_path(cur_seg) * 0.90: return
        tail = self.mandatory_route([nm] + remaining[1:]) if remaining[1:] else []
        suffix = best + (tail[1:] if tail else [])
        if suffix == t.path[t.edge_index:]: return
        t.path = t.path[:t.edge_index] + suffix
        t.progress = 0; t.reroute_count += 1; t.last_reroute = frame; t.reroute_flash = 25
        self.update_occupancy()

    def step(self, frame):
        self.update_occupancy(); self.update_signals(); self.detect_headon()
        risks = []
        for t in self.trains:
            t.update(); risks.append(self.compute_risk(t))
        if self.enable_reroute:
            self.reroute_one(frame)
        # metrics
        if len(self.trains) > 1:
            min_hw = min(np.linalg.norm(np.array(self.train_pos(a)) - np.array(self.train_pos(b)))
                         for i, a in enumerate(self.trains) for b in self.trains[i + 1:])
        else:
            min_hw = 9.99
        self.min_headway_hist.append(min_hw)
        self.avg_risk_hist.append(float(np.mean(risks)) if risks else 0.0)
        stopped = {t.train_id for t in self.trains if t.stopped_headon}
        self.headon_events += len(stopped - self._prev_stopped)
        self._prev_stopped = stopped
        return risks, min_hw

    def metrics(self):
        hw = np.array(self.min_headway_hist) if self.min_headway_hist else np.array([0.0])
        return {
            "frames": len(self.min_headway_hist),
            "trains": len(self.trains),
            "stations": self.G.number_of_nodes(),
            "segments": self.G.number_of_edges(),
            "total_reroutes": int(sum(t.reroute_count for t in self.trains)),
            "reroutes_per_train": {t.train_id: t.reroute_count for t in self.trains},
            "headon_events": int(self.headon_events),
            "min_headway_overall": round(float(hw.min()), 3),
            "mean_min_headway": round(float(hw.mean()), 3),
            "critical_frac": round(float((hw < CRITICAL_DIST).mean()), 4),
            "throughput_edges": int(sum(t.completed_edges for t in self.trains)),
            "avg_risk": round(float(np.mean(self.avg_risk_hist)) if self.avg_risk_hist else 0, 4),
        }


# ════════════════════════════════════════════════════════════════════════
# 5. HEADLESS METRICS + REROUTE COMPARISON
# ════════════════════════════════════════════════════════════════════════
def run_headless(network, train_specs, frames, model, scaler, threshold,
                 enable_reroute=True, seed=0):
    sim = RailwaySim(network, train_specs, model, scaler, threshold,
                     seed=seed, enable_reroute=enable_reroute)
    for f in range(frames):
        sim.step(f)
    return sim.metrics()


def compare_rerouting(network, train_specs, frames, model, scaler, threshold):
    on  = run_headless(network, train_specs, frames, model, scaler, threshold, True, seed=0)
    off = run_headless(network, train_specs, frames, model, scaler, threshold, False, seed=0)
    return on, off


# ════════════════════════════════════════════════════════════════════════
# 6. DASHBOARD VIDEO RENDERER
# ════════════════════════════════════════════════════════════════════════
def render_video(network, train_specs, frames, model, scaler, threshold,
                 out_path, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    import networkx as nx

    sim = RailwaySim(network, train_specs, model, scaler, threshold,
                     seed=seed, enable_reroute=True)
    SIG = {0: "#00E676", 1: "#FFD740", 2: "#FF1744"}
    load_cmap = plt.cm.RdYlGn_r

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), dpi=96)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    ax_net  = fig.add_subplot(gs[:, :2])
    ax_hw   = fig.add_subplot(gs[0, 2])
    ax_rr   = fig.add_subplot(gs[1, 2])

    nx.draw_networkx_edges(sim.G, sim.pos, ax=ax_net, edge_color="#37474F", width=6, alpha=0.6)
    nx.draw_networkx_nodes(sim.G, sim.pos, ax=ax_net, nodelist=list(sim.majors),
                           node_color="#FFD700", node_size=1600, node_shape="s")
    nx.draw_networkx_nodes(sim.G, sim.pos, ax=ax_net, nodelist=list(sim.minors),
                           node_color="#546E7A", node_size=450)
    nx.draw_networkx_labels(sim.G, sim.pos, ax=ax_net, font_color="white",
                            font_weight="bold", font_size=10)
    edge_lines = {}
    for (u, v) in sim.SEGMENTS:
        x1, y1 = sim.pos[u]; x2, y2 = sim.pos[v]
        ln, = ax_net.plot([x1, x2], [y1, y2], lw=8, alpha=0, zorder=2, solid_capstyle="round")
        edge_lines[(u, v)] = ln
    signal_dots = {st: ax_net.scatter(*[[c] for c in (sim.pos[st][0], sim.pos[st][1] + 0.55)],
                                      s=180, zorder=10, color=SIG[0],
                                      edgecolors="white", linewidths=1.2) for st in sim.majors}
    n = len(sim.trains)
    train_scat = ax_net.scatter([], [], s=420, zorder=6, edgecolors="white", linewidths=2)
    brake_c  = [Circle((0, 0), 0.34, fill=False, ec="#FF1744", lw=3, visible=False, zorder=8) for _ in range(n)]
    rr_c     = [Circle((0, 0), 0.50, fill=False, ec="#00E5FF", lw=2.5, ls="--", visible=False, zorder=8) for _ in range(n)]
    ho_c     = [Circle((0, 0), 0.60, fill=False, ec="#FFFFFF", lw=3, visible=False, zorder=8) for _ in range(n)]
    for c in brake_c + rr_c + ho_c: ax_net.add_patch(c)
    tlabels = [ax_net.text(0, 0, "", fontsize=8, color="white", ha="center",
                           va="bottom", fontweight="bold", zorder=9) for _ in range(n)]
    xs = [p[0] for p in sim.pos.values()]; ys = [p[1] for p in sim.pos.values()]
    ax_net.set_xlim(min(xs) - 1.5, max(xs) + 2); ax_net.set_ylim(min(ys) - 1.5, max(ys) + 2)
    ax_net.axis("off"); ax_net.set_title("Railway Network — risk / braking / rerouting",
                                         fontsize=12, fontweight="bold")
    stat_txt = ax_net.text(0.01, 0.99, "", transform=ax_net.transAxes, va="top",
                           ha="left", fontsize=9, family="monospace", color="white",
                           bbox=dict(boxstyle="round", fc="#101820", alpha=0.7))

    hw_line, = ax_hw.plot([], [], "cyan", lw=2)
    ax_hw.axhline(CRITICAL_DIST, color="red", ls="--", lw=1.5)
    ax_hw.axhline(WARNING_DIST, color="orange", ls="--", lw=1.5)
    ax_hw.set_title("Min headway", fontsize=10); ax_hw.set_ylim(0, 8); ax_hw.grid(alpha=0.25)
    rr_bars = ax_rr.barh(range(n), [0] * n, color="#29B6F6")
    ax_rr.set_yticks(range(n)); ax_rr.set_yticklabels([t.train_id for t in sim.trains], fontsize=8)
    ax_rr.set_title("Reroute events", fontsize=10); ax_rr.set_xlim(0, 10); ax_rr.grid(alpha=0.25, axis="x")

    def frame_update(frame):
        risks, min_hw = sim.step(frame)
        for seg, ln in edge_lines.items():
            load = sim.seg_pred[seg]; ln.set_color(load_cmap(load)); ln.set_alpha(0.15 + 0.75 * load)
        for st, dot in signal_dots.items():
            dot.set_color(SIG[sim.station_signal[st]])
        px, py, cols = [], [], []
        for i, t in enumerate(sim.trains):
            x, y = sim.train_pos(t); px.append(x); py.append(y)
            ho_c[i].set_center((x, y)); ho_c[i].set_visible(t.stopped_headon)
            brake_c[i].set_center((x, y)); brake_c[i].set_visible(t.is_braking and not t.stopped_headon)
            rr_c[i].set_center((x, y)); rr_c[i].set_visible(t.reroute_flash > 0)
            r = t.collision_risk
            c = "#FFFFFF" if t.stopped_headon else "#FF1744" if r > threshold else \
                "#FF9800" if r > 0.40 else "#FFEB3B" if r > 0.20 else "#4CAF50"
            cols.append(c)
            sym = "*" if t.priority == 1 else "o" if t.priority == 2 else "."
            tlabels[i].set_position((x, y + 0.45)); tlabels[i].set_text(f"{t.train_id}{sym}")
        train_scat.set_offsets(np.c_[px, py]); train_scat.set_color(cols)
        h = sim.min_headway_hist
        hw_line.set_data(range(len(h)), h); ax_hw.set_xlim(0, max(50, len(h)))
        maxr = max((t.reroute_count for t in sim.trains), default=0)
        ax_rr.set_xlim(0, max(10, maxr + 2))
        for bar, t in zip(rr_bars, sim.trains): bar.set_width(t.reroute_count)
        stat_txt.set_text(
            f"frame {frame}\ntrains {n}  reroutes {sum(t.reroute_count for t in sim.trains)}\n"
            f"head-on stops {sim.headon_events}\nmin headway {min_hw:.2f}\n"
            f"avg risk {np.mean(risks) if risks else 0:.3f}")
        return [train_scat]

    ani = FuncAnimation(fig, frame_update, frames=frames, interval=50, blit=False)
    ani.save(out_path, writer="ffmpeg", fps=20, dpi=96)
    plt.close(fig)
    return out_path, sim.metrics()


# ════════════════════════════════════════════════════════════════════════
# 7. GRADIO UI
# ════════════════════════════════════════════════════════════════════════
def _fmt_metrics(m, title):
    rr = ", ".join(f"{k}:{v}" for k, v in m["reroutes_per_train"].items())
    return (f"### {title}\n"
            f"- Trains / stations / segments: **{m['trains']} / {m['stations']} / {m['segments']}**\n"
            f"- Frames simulated: **{m['frames']}**\n"
            f"- **Total reroutes:** {m['total_reroutes']}  ({rr})\n"
            f"- **Head-on stops (collisions averted):** {m['headon_events']}\n"
            f"- Min headway (overall / mean): **{m['min_headway_overall']} / {m['mean_min_headway']}**\n"
            f"- Time below critical headway: **{m['critical_frac']*100:.1f}%**\n"
            f"- Throughput (edges completed): **{m['throughput_edges']}**\n"
            f"- Mean collision risk: **{m['avg_risk']}**\n")


def build_ui(model, scaler, cfg):
    import gradio as gr
    threshold = cfg.get("optimal_threshold", 0.5)
    test_line = (f"Model: TEST P={cfg.get('test_precision','?')} "
                 f"R={cfg.get('test_recall','?')} F1={cfg.get('test_f1','?')} "
                 f"| threshold={threshold:.3f} | device={DEVICE}")

    def add_train(trains_json, priority):
        try:
            trains = json.loads(trains_json)
        except Exception as e:
            return trains_json, f"JSON error: {e}"
        majors = json.loads(DEFAULT_NETWORK_JSON)["majors"]
        nid = f"T{len(trains)+1}"
        route = random.sample(majors, k=min(3, len(majors)))
        spec = {"id": nid, "route": route, "priority": int(priority),
                "speed": [0.009 + 0.002 * (3 - int(priority)), 0.014 + 0.002 * (3 - int(priority))]}
        trains.append(spec)
        return json.dumps(trains, indent=2), f"Added {nid} route={route} priority={priority}"

    def remove_train(trains_json):
        try:
            trains = json.loads(trains_json)
        except Exception as e:
            return trains_json, f"JSON error: {e}"
        if trains:
            removed = trains.pop()
            return json.dumps(trains, indent=2), f"Removed {removed['id']}"
        return trains_json, "No trains to remove"

    def reset_cfg():
        return DEFAULT_NETWORK_JSON, DEFAULT_TRAINS_JSON, "Reset to defaults"

    def run_metrics(network_json, trains_json, frames):
        try:
            network = json.loads(network_json); trains = json.loads(trains_json)
        except Exception as e:
            return f"**Config JSON error:** {e}", None
        on, off = compare_rerouting(network, trains, int(frames), model, scaler, threshold)
        delta_hw = round(on["mean_min_headway"] - off["mean_min_headway"], 3)
        delta_ho = off["headon_events"] - on["headon_events"]
        summary = (f"## Rerouting performance evaluation\n"
                   f"_{test_line}_\n\n"
                   + _fmt_metrics(on, "WITH rerouting (Dijkstra + congestion)")
                   + "\n" + _fmt_metrics(off, "WITHOUT rerouting (baseline)")
                   + f"\n### Effect of rerouting\n"
                   f"- Mean min-headway change: **{delta_hw:+.3f}** (higher = safer spacing)\n"
                   f"- Head-on stops avoided vs baseline: **{delta_ho:+d}**\n")
        return summary, None

    def run_video(network_json, trains_json, frames):
        try:
            network = json.loads(network_json); trains = json.loads(trains_json)
        except Exception as e:
            return None, f"**Config JSON error:** {e}"
        out = os.path.join(HERE, "railway_ui_render.mp4")
        path, m = render_video(network, trains, int(frames), model, scaler, threshold, out)
        return path, _fmt_metrics(m, "This run (with rerouting)")

    with gr.Blocks(title="Railway Safety Framework") as demo:
        gr.Markdown(f"# 🚆 Railway Safety Framework — interactive scale-up\n{test_line}")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Trains  (edit JSON or use buttons)")
                trains_box = gr.Code(value=DEFAULT_TRAINS_JSON, language="json", lines=16,
                                     label="Trains")
                with gr.Row():
                    prio = gr.Dropdown([1, 2, 3], value=3, label="New train priority")
                    add_btn = gr.Button("➕ Add train")
                    rm_btn = gr.Button("➖ Remove last")
                gr.Markdown("### Network  (nodes / majors / weighted edges)")
                net_box = gr.Code(value=DEFAULT_NETWORK_JSON, language="json", lines=14,
                                  label="Network")
                reset_btn = gr.Button("↺ Reset to defaults")
                status = gr.Markdown("")
            with gr.Column(scale=1):
                frames = gr.Slider(100, 1200, value=400, step=50, label="Frames to simulate")
                with gr.Row():
                    metrics_btn = gr.Button("📊 Evaluate (fast, no video)", variant="primary")
                    video_btn = gr.Button("🎬 Render video")
                metrics_md = gr.Markdown("")
                video_out = gr.Video(label="Simulation", autoplay=True)

        add_btn.click(add_train, [trains_box, prio], [trains_box, status])
        rm_btn.click(remove_train, [trains_box], [trains_box, status])
        reset_btn.click(reset_cfg, None, [net_box, trains_box, status])
        metrics_btn.click(run_metrics, [net_box, trains_box, frames], [metrics_md, video_out])
        video_btn.click(run_video, [net_box, trains_box, frames], [video_out, metrics_md])

    return demo


DEFAULT_NETWORK_JSON = json.dumps(DEFAULT_NETWORK, indent=2)
DEFAULT_TRAINS_JSON  = json.dumps(DEFAULT_TRAINS, indent=2)


# ════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="force retrain the model")
    ap.add_argument("--no-ui", action="store_true", help="build artifacts only, no UI")
    ap.add_argument("--share", action="store_true", help="gradio public share link")
    args = ap.parse_args()

    if args.train:
        for p in (MODEL_PATH, CONFIG_PATH):
            if os.path.exists(p): os.remove(p)
        model, scaler, cfg = train_pipeline()
    else:
        print("Loading model artifacts (training if missing) ...")
        model, scaler, cfg = load_model()
    print(f"Ready. {cfg}")

    if args.no_ui:
        return
    demo = build_ui(model, scaler, cfg)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
