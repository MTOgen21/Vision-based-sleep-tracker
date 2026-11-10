"""Lying position (back, left side, right side, front) and roll-overs from keypoints."""
import numpy as np

from .pose import L_HIP, L_SHOULDER, MIN_SCORE, NOSE, R_HIP, R_SHOULDER, TORSO

POSITIONS = ['back', 'left', 'right', 'front']
UNKNOWN = ''
# Shoulder width as a fraction of torso length. Seen flat-on it is about 0.7;
# lying on the side stacks the shoulders and it collapses towards zero.
# Starting values: tune against overlay images from real recordings.
SIDE_BELOW = 0.35
VOTE_S = 10.0       # each label is the majority over this window
MIN_DWELL_S = 10.0  # shorter stretches are treated as noise
ROLL_WINDOW_S = 10.0


def features(points, scores):
    """(signed shoulder width, sideways nose offset), both relative to torso length.

    The sign convention is the camera's view of someone lying on their back:
    their left shoulder is on the positive side.
    """
    if scores[TORSO].min() < MIN_SCORE:
        return np.nan, np.nan
    shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
    spine = shoulders - (points[L_HIP] + points[R_HIP]) / 2
    length = np.hypot(*spine)
    if length < 1:
        return np.nan, np.nan
    head = spine / length
    across = np.array([-head[1], head[0]])
    width = (points[L_SHOULDER] - points[R_SHOULDER]) @ across / length
    face = (points[NOSE] - shoulders) @ across / length if scores[NOSE] >= MIN_SCORE else np.nan
    return width, face


def classify(width, face):
    if np.isnan(width):
        return UNKNOWN
    if abs(width) >= SIDE_BELOW:
        # Facing the camera the left shoulder is on the positive side; face down it flips.
        # 'front' is the least reliable label: pose models often mix up left and right there.
        return 'back' if width > 0 else 'front'
    # On the side, the nose points the way the chest faces. Rolling onto the left
    # side turns the chest towards the subject's left, the positive direction.
    direction = face if not np.isnan(face) else width
    return 'left' if direction > 0 else 'right'


def timeline(times, labels):
    """Smooth per-frame labels into [(start, end, position)] stretches."""
    times = np.asarray(times)
    known = np.array([label != UNKNOWN for label in labels])
    if not known.any():
        return []
    times, labels = times[known], [l for l, k in zip(labels, known) if k]
    codes = np.array([POSITIONS.index(l) for l in labels])
    counts = np.vstack([np.zeros(len(POSITIONS), int), np.cumsum(np.eye(len(POSITIONS), dtype=int)[codes], 0)])
    first = np.searchsorted(times, times - VOTE_S / 2)
    last = np.searchsorted(times, times + VOTE_S / 2, side='right')
    voted = (counts[last] - counts[first]).argmax(1)
    stretches = []
    start = 0
    for i in range(1, len(voted) + 1):
        if i == len(voted) or voted[i] != voted[start]:
            stretches.append([times[start], times[i - 1], POSITIONS[voted[start]]])
            start = i
    # Drop brief flickers, then join neighbours that now agree.
    merged = []
    for stretch in stretches:
        if merged and (stretch[1] - stretch[0] < MIN_DWELL_S or merged[-1][2] == stretch[2]):
            merged[-1][1] = stretch[1]
        else:
            merged.append(stretch)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_DWELL_S:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [tuple(stretch) for stretch in merged]


def roll_overs(stretches, times, active):
    """Position changes that coincide with movement: [(time, from, to)]."""
    times, active = np.asarray(times), np.asarray(active)
    events = []
    for before, after in zip(stretches, stretches[1:]):
        if before[2] == after[2]:
            continue
        moment = (before[1] + after[0]) / 2
        if active[np.abs(times - moment) <= ROLL_WINDOW_S].any():
            events.append((moment, before[2], after[2]))
    return events
