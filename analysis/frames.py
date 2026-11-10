"""Load a recorded session and put every frame on one clock."""
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
CALIBRATION = OUTPUT / 'calibration.json'
# A longer silence than this is missing data, not stillness.
MAX_GAP_S = 1.0


@dataclass
class Session:
    name: str
    files: list        # one Path per frame, in the order received
    sequence: np.ndarray
    time: np.ndarray   # UTC seconds on the camera's clock


def load_session(path):
    """Read a session folder written by tools/save_jpegs.py."""
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    with (folder / 'frames.csv').open(newline='', encoding='utf-8') as log:
        rows = list(csv.DictReader(log))
    if not rows:
        raise SystemExit(f'No frames recorded in {folder}')
    received = np.array([datetime.fromisoformat(r['received_utc']).timestamp() for r in rows])
    uptime = np.array([int(r['camera_uptime_us']) for r in rows]) / 1e6
    # The camera clock restarts from zero on every reboot, so each boot gets its
    # own offset to UTC. The median ignores frames that were delayed in transit.
    boot = np.concatenate([[0], np.cumsum(np.diff(uptime) < 0)])
    time = np.empty(len(rows))
    for b in np.unique(boot):
        mask = boot == b
        time[mask] = uptime[mask] + np.median(received[mask] - uptime[mask])
    return Session(folder.name, [folder / r['filename'] for r in rows],
                   np.array([int(r['sequence']) for r in rows]), time)


def load_calibration(path, frame_size):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f'No calibration at {path}. Run: py -m analysis.calibrate <session>')
    calibration = json.loads(path.read_text(encoding='utf-8'))
    if tuple(calibration['frame_size']) != tuple(frame_size):
        raise SystemExit('Calibration was made at a different frame size. Calibrate again.')
    return calibration
