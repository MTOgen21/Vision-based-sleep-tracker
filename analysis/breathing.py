"""Breathing waveform, rate and pauses from the torso signals in motion.py.

The body has to be still for the faint breathing signal to be readable, so the
recording is split into still runs and each run is analysed on its own.
"""
from dataclasses import dataclass

import numpy as np

from .frames import MAX_GAP_S
from .motion import ACTIVE_FRACTION, GRID

FS = 5.0                    # signals are resampled to this uniform rate, Hz
BAND = (8 / 60, 39 / 60)    # plausible breathing: 8 to 39 breaths/min
TAPER_HZ = 0.03             # soft filter edges, to limit ringing around pauses
PEAK_HZ = 0.05
MIN_RUN_S = 30.0
WINDOW_S = 60.0
SETTLE_S = 1.0              # ignored either side of any movement
KEEP = 12                   # torso signals kept: the ones that look most periodic
# Starting values: tune against paced-breathing sessions.
# Pure noise scores up to about 0.7 here, because the most periodic-looking cells are chosen.
GOOD_QUALITY = 0.7
PAUSE_FRACTION = 0.3        # a pause is the waveform falling below this share of normal
PAUSE_MIN_S = 10.0


@dataclass
class Run:
    time: np.ndarray        # uniform FS grid, UTC seconds
    wave: np.ndarray        # breathing waveform, arbitrary units
    envelope: np.ndarray    # breathing depth
    baseline: np.ndarray    # typical depth around each moment
    quality: float          # median spectral quality across the run


def bandpass(x):
    """Remove the trend, then keep only the breathing band. Works down columns."""
    n = len(x)
    index = np.arange(n)
    slope, offset = np.polyfit(index, x, 1)
    x = x - (np.multiply.outer(index, slope) + offset)
    freq = np.fft.rfftfreq(n, 1 / FS)
    ramp = np.clip((freq - BAND[0]) / TAPER_HZ + 1, 0, 1) * np.clip((BAND[1] - freq) / TAPER_HZ + 1, 0, 1)
    gain = 0.5 - 0.5 * np.cos(np.pi * ramp)
    spectrum = np.fft.rfft(x, axis=0)
    return np.fft.irfft((spectrum.T * gain).T, n, axis=0)


def peak(x):
    """(frequency in Hz, quality) of the strongest breathing-band rhythm in x.

    Quality is the share of in-band power sitting at that one frequency: near 1
    for clean regular breathing, around 0.2 to 0.3 for noise.
    """
    size = max(4096, 1 << int(np.ceil(np.log2(len(x) * 4))))
    power = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), size)) ** 2
    freq = np.fft.rfftfreq(size, 1 / FS)
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    total = power[band].sum()
    if total <= 0:
        return np.nan, 0.0
    top = freq[band][power[band].argmax()]
    return top, float(power[band & (np.abs(freq - top) <= PEAK_HZ)].sum() / total)


def _windows(n):
    size, step = int(WINDOW_S * FS), int(WINDOW_S * FS / 2)
    if n <= size:
        return [slice(0, n)]
    return [slice(start, start + size) for start in range(0, n - size + 1, step)]


def _envelope(wave):
    """Instantaneous breathing depth (Hilbert envelope), lightly smoothed."""
    n = len(wave)
    weights = np.zeros(n)
    weights[0] = 1
    weights[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        weights[n // 2] = 1
    depth = np.abs(np.fft.ifft(np.fft.fft(wave) * weights))
    size = int(2 * FS)
    return np.convolve(np.pad(depth, size // 2, mode='edge'), np.ones(size) / size, 'same')[size // 2:-(size // 2)]


def _baseline(envelope):
    """Rolling two-minute median of the depth, computed at 1 Hz."""
    step, half = int(FS), 60
    coarse = envelope[::step]
    padded = np.pad(coarse, half, mode='reflect' if len(coarse) > half else 'edge')
    rolled = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    return np.interp(np.arange(len(envelope)), np.arange(len(coarse)) * step, rolled)


def _run(time, signals, reference):
    grid = np.arange(time[0], time[-1], 1 / FS)
    cells = GRID * GRID
    # Velocity summed over time is displacement, which is what breathing looks like.
    step = np.diff(time, prepend=time[0])
    moved = np.cumsum(np.nan_to_num(signals[:, cells:]) * step[:, None], axis=0)
    raw = np.column_stack([signals[:, :cells], moved])
    x = bandpass(np.column_stack([np.interp(grid, time, column) for column in raw.T]))
    # Room-light changes hit the surroundings too, so remove whatever part of each
    # brightness signal follows the surroundings.
    room = bandpass(np.interp(grid, time, reference))
    if room @ room > 0:
        x[:, :cells] -= np.outer(room, room @ x[:, :cells] / (room @ room))
    spread = x.std(axis=0)
    x = x[:, spread > 1e-9] / spread[spread > 1e-9]
    if x.shape[1] == 0:
        return None
    # Most cells carry only noise. Keep the ones with the clearest rhythm, then
    # take their shared component as the breathing waveform.
    windows = _windows(len(x))
    rhythm = np.array([np.mean([peak(column[w])[1] for w in windows]) for column in x.T])
    chosen = x[:, np.argsort(rhythm)[-KEEP:]]
    left, strength, _ = np.linalg.svd(chosen, full_matrices=False)
    wave = left[:, 0] * strength[0] / np.sqrt(chosen.shape[1])
    envelope = _envelope(wave)
    quality = float(np.median([peak(wave[w])[1] for w in windows]))
    return Run(grid, wave, envelope, _baseline(envelope), quality)


def analyse(time, signals, reference, torso_moving, region):
    """Split the recording into still runs and extract a waveform from each."""
    time = np.asarray(time)
    busy = ~(np.asarray(torso_moving) < ACTIVE_FRACTION)   # NaN (no flow) counts as busy
    busy_times = time[busy]
    if len(busy_times):
        nearest = np.searchsorted(busy_times, time)
        before = np.abs(time - busy_times[np.clip(nearest - 1, 0, len(busy_times) - 1)])
        after = np.abs(time - busy_times[np.clip(nearest, 0, len(busy_times) - 1)])
        busy = np.minimum(before, after) <= SETTLE_S
    runs = []
    start = None
    for i in range(len(time) + 1):
        ends = i == len(time) or busy[i] or (
            start is not None and (time[i] - time[i - 1] > MAX_GAP_S or region[i] != region[start]))
        if start is not None and ends:
            if time[i - 1] - time[start] >= MIN_RUN_S:
                run = _run(time[start:i], signals[start:i], reference[start:i])
                if run is not None:
                    runs.append(run)
            start = None
        if start is None and i < len(time) and not busy[i]:
            start = i
    return runs


def rate_between(runs, start, end):
    """(breaths/min, quality) over [start, end], or (nan, 0) without enough still data."""
    best = None
    for run in runs:
        inside = (run.time >= start) & (run.time <= end)
        if inside.sum() >= MIN_RUN_S * FS and (best is None or inside.sum() > best[1].sum()):
            best = run, inside
    if best is None:
        return np.nan, 0.0
    frequency, quality = peak(best[0].wave[best[1]])
    return frequency * 60, quality


def pauses(run):
    """[(start, end)] where breathing depth collapses for PAUSE_MIN_S or longer.

    Only attempted when the run's breathing signal is clear; otherwise low depth
    just means there was never a readable signal.
    """
    if run.quality < GOOD_QUALITY:
        return []
    low = run.envelope < PAUSE_FRACTION * run.baseline
    edge = int(3 * FS)   # the filter is unreliable right at the ends of a run
    low[:edge] = low[-edge:] = False
    found = []
    start = None
    for i in range(len(low) + 1):
        if i < len(low) and low[i]:
            start = i if start is None else start
        elif start is not None:
            if (i - start) / FS >= PAUSE_MIN_S:
                found.append((run.time[start], run.time[i - 1]))
            start = None
    return found
