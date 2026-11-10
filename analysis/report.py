"""One-page picture of a session: movement, breathing, position and data quality."""
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .breathing import GOOD_QUALITY
from .position import POSITIONS

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
CRITICAL = '#d03b3b'
POSITION_COLOURS = dict(zip(POSITIONS, [BLUE, ORANGE, AQUA, YELLOW]))
POSITION_NAMES = {'back': 'Back', 'left': 'Left side', 'right': 'Right side', 'front': 'Front'}


def summary(path, name, times, motion, epochs, runs, pauses, stretches, rolls, paced):
    start = times[0]
    hours = times[-1] - start > 3 * 3600
    scale = 3600 if hours else 60

    def x(seconds):
        return (np.asarray(seconds) - start) / scale

    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9, 'text.color': INK,
                         'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'xtick.labelcolor': SECONDARY, 'ytick.labelcolor': SECONDARY})
    figure, axes = plt.subplots(5, 1, figsize=(11, 10.5), sharex=True, facecolor=SURFACE,
                                gridspec_kw={'height_ratios': [2, 2.4, 2, 1.5, 1.4], 'hspace': 0.5})
    for axis in axes:
        axis.set_facecolor(SURFACE)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3)

    def title(axis, text, note=''):
        axis.set_title(text, loc='left', fontsize=10, fontweight='bold', color=INK, pad=6)
        if note:
            axis.set_title(note, loc='right', fontsize=8.5, color=SECONDARY, pad=6)

    # Movement. Gaps in the recording break the line instead of being bridged.
    axis = axes[0]
    gapped = np.where(np.isnan(motion), np.nan, motion)
    axis.plot(x(times), gapped, color=BLUE, linewidth=1.2)
    for moment, _, _ in rolls:
        axis.axvline(x(moment), color=SECONDARY, linewidth=1, linestyle=(0, (3, 2)))
    axis.set_ylim(bottom=0)
    axis.set_ylabel('pixels / s')
    title(axis, 'Movement', f'{len(rolls)} roll-over{"" if len(rolls) == 1 else "s"} (dashed lines)')

    # Breathing rate per 30 s epoch.
    axis = axes[1]
    centre = np.array([e['time'] for e in epochs])
    rate = np.array([e['breaths_per_min'] for e in epochs])
    quality = np.array([e['breathing_quality'] for e in epochs])
    good = quality >= GOOD_QUALITY
    axis.plot(x(centre[~good]), rate[~good], 'o', markersize=5, markerfacecolor=SURFACE,
              markeredgecolor=MUTED, markeredgewidth=1, linestyle='none', label='unclear signal')
    axis.plot(x(centre[good]), rate[good], 'o', markersize=5.5, color=BLUE,
              markeredgecolor=SURFACE, markeredgewidth=0.8, linestyle='none', label='clear signal')
    labelled = False
    for begin, finish, kind, _, target, _, _ in paced:
        if kind == 'paced':
            axis.plot(x([begin, finish]), [target, target], color=INK, linewidth=2, solid_capstyle='butt',
                      label=None if labelled else 'paced target')
            axis.annotate(f'{target:.0f}', (x(finish), target), xytext=(4, 0), textcoords='offset points',
                          va='center', fontsize=8.5, color=SECONDARY)
            labelled = True
    shown = rate[~np.isnan(rate)]
    if len(shown):
        # Headroom keeps the legend clear of the data.
        axis.set_ylim(max(0, min(shown.min(), 8) - 2), max(shown.max(), 24) + 8)
    axis.set_ylabel('breaths / min')
    axis.legend(loc='upper right', ncols=3, frameon=False, fontsize=8.5, labelcolor=SECONDARY,
                handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    median = np.median(rate[good]) if good.any() else np.nan
    title(axis, 'Breathing rate', f'median {median:.1f} breaths/min when clear' if good.any() else 'no clear signal')

    # Breathing depth relative to the surrounding two minutes.
    axis = axes[2]
    for run in runs:
        axis.plot(x(run.time), run.envelope / np.maximum(run.baseline, 1e-12), color=BLUE, linewidth=1.2)
    for begin, finish in pauses:
        axis.axvspan(x(begin), x(finish), color=CRITICAL, alpha=0.18, linewidth=0)
        axis.annotate(f'pause {finish - begin:.0f} s', (x((begin + finish) / 2), 1.9), ha='center',
                      va='top', fontsize=8.5, color=INK)
    axis.axhline(1, color=AXIS, linewidth=0.8)
    axis.set_ylim(0, 2)
    axis.set_ylabel('depth (1 = usual)')
    title(axis, 'Breathing depth while still',
          f'{len(pauses)} pause{"" if len(pauses) == 1 else "s"} of 10 s or more (shaded)')

    # Position: one row per position, so colour is never the only cue.
    axis = axes[3]
    for begin, finish, label in stretches:
        row = POSITIONS.index(label)
        axis.barh(row, x(finish) - x(begin), left=x(begin), height=0.62, color=POSITION_COLOURS[label],
                  edgecolor=SURFACE, linewidth=1)
    axis.set_yticks(range(len(POSITIONS)), [POSITION_NAMES[p] for p in POSITIONS])
    axis.set_ylim(len(POSITIONS) - 0.5, -0.5)
    axis.grid(False)
    known = sum(b - a for a, b, _ in stretches)
    shares = [f'{POSITION_NAMES[p].lower()} {sum(b - a for a, b, q in stretches if q == p) / known:.0%}'
              for p in POSITIONS if any(q == p for _, _, q in stretches)] if known else []
    title(axis, 'Position', ', '.join(shares) if shares else 'no pose found')

    # Data quality: both measures run from 0 to 1.
    axis = axes[4]
    axis.plot(x(centre), [e['coverage'] for e in epochs], color=BLUE, linewidth=1.6, label='frames received')
    axis.plot(x(centre), [e['keypoint_score'] for e in epochs], color=ORANGE, linewidth=1.6,
              label='pose confidence')
    axis.set_ylim(0, 1.45)
    axis.set_yticks([0, 0.5, 1])
    axis.set_ylabel('share / score')
    axis.legend(loc='upper right', ncols=2, frameon=False, fontsize=8.5, labelcolor=SECONDARY, borderaxespad=0)
    axis.set_xlabel(f'{"hours" if hours else "minutes"} since {datetime.fromtimestamp(start):%H:%M:%S} local time')
    axis.set_xlim(0, x(times[-1]))
    title(axis, 'Data quality')

    figure.suptitle(f'Session {name}', x=0.07, y=0.965, ha='left', fontsize=13, fontweight='bold')
    figure.text(0.07, 0.94, f'{datetime.fromtimestamp(start):%d %b %Y}, '
                f'{(times[-1] - start) / 60:.0f} min recorded', fontsize=9.5, color=SECONDARY)
    figure.text(0.07, 0.022, 'A pause is a stretch with no readable breathing movement while the body is still. '
                'It is a flag for interest, not a medical measurement.', fontsize=8.5, color=SECONDARY)
    figure.subplots_adjust(left=0.09, right=0.965, top=0.9, bottom=0.095)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
