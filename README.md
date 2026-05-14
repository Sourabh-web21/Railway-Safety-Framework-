# Railway Train Monitoring System

Real-time train collision avoidance with ML risk prediction, dynamic rerouting, traffic signals, and comprehensive dashboard monitoring.

## Features

| Feature | Status |
|---------|--------|
| Traffic Signals at 4 major stations | Live |
| Head-on Collision Detection (visual stops, no teleport) | Active |
| Dynamic Rerouting at major stations only | Priority-based |
| Track Occupancy Heatmap (40-frame history) | Real-time |
| Brake Intensity Visualization | Per-train |
| ML Collision Risk Model (PyTorch LSTM) | Sequence-based |
| Live Dashboard (risk, speed, headway, stats) | 1200-frame animation |

## Network Architecture
W ←→ C ←→ E (Main Northern Corridor - Fastest: 3 units/segment)
↓ ↓ ↓
WS←→S←→SE (Southern Arc - Medium: 4 units/segment)
↑ ↑ ↑
NW←→NE (Emergency Northern Spurs - Slowest: 5 units/segment)



4 Major Stations (W, C, E, S) + 7 Minor Waypoints

## Train Operations

| Train | Type | Priority | Route | Speed Range |
|-------|------|----------|-------|-------------|
| T1 | Express | 1 (Never reroutes) | W→C→E | 1.3-1.8 |
| T2 | Regional | 2 | E→C→W | 1.1-1.6 |
| T3 | Regional | 2 | S→C→W | 1.1-1.6 |
| T4 | Regional | 2 | W→S→E | 1.0-1.5 |
| T5 | Freight | 3 | C→W→S | 0.9-1.4 |
| T6 | Freight | 3 | C→S→E | 0.9-1.4 |

## ML Risk Prediction Model

Input Features (8D per frame):
velocity, acceleration, headway, gradient, curvature, speed_limit, signal_state, friction


LSTM Sequence Processing (10-frame window) → Risk Score (0-1)

Risk-based Actions:
- Risk > 0.5 → Red trains + emergency braking
- Risk > 0.40 → Orange + warning braking  
- Risk > 0.20 → Yellow + light braking
- Risk < 0.20 → Green + normal speed

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/railway-train-monitoring.git
cd railway-train-monitoring
pip install torch networkx matplotlib numpy
jupyter notebook Railway_Train_Monitoring.ipynb
```

## Technical Implementation
Core: NetworkX (routing) + PyTorch (ML) + Matplotlib (animation)
Routing: Dijkstra + congestion penalties (15× predicted load)
Signals: Distance-based (100m=red, 250m=yellow)
Rerouting: Sequential (1 train/frame, freight first)
Collision: Headway <1.2 grid units → 60-frame stop
Animation: 1200 frames @ 20fps = 60 seconds MP4


## Learning Outcomes

- Graph Algorithms: Dijkstra routing with dynamic costs
- ML Integration: LSTM risk prediction in real-time simulation  
- Visualization: Multi-panel animated dashboard
- Collision Avoidance: Head-on detection + priority rerouting
- Traffic Systems: Signal logic + occupancy prediction
