# Factory worker activity

Tracks people in factory-floor video and shows idle/active times, heatmaps in the factory floor, and total movement.

## Installation

```bash
pip install -r requirements.txt
```

Put your input **`.mp4`** under `videos/` (name matches `INPUT_VIDEO_NAME` in `config.py`).

## Run

```bash
python main.py
```

Outputs go to `output/` (including `stats.json` at the project root). Then:

```bash
python server.py
```

Open **http://127.0.0.1:5000** for the dashboard.
