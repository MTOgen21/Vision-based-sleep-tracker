"""Analyse one recorded session: py -m analysis.run raw_image/<session>"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import breathing, position, report
from .frames import CALIBRATION, MAX_GAP_S, OUTPUT, load_calibration, load_session
from .motion import ACTIVE_FRACTION, Motion
from .pose import NAMES, UPPER, PoseModel, Tracker, draw, overlap, torso_box

EPOCH_S = 30.0
REGION_KEEP = 0.7    # the torso region only moves when the new one overlaps less than this
PACE_SKIP_S = 10.0   # time allowed to settle into each paced rate


def read_frame(path, size=None):
    data = np.fromfile(path, np.uint8) if path.exists() else np.empty(0, np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if frame is None or (size and frame.shape[:2] != size):
        return None
    return frame


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec='milliseconds')


def load_pace(paths, start, end):
    """Paced-breathing stages (from tools/pace_breathing.py) that fall inside the session."""
    stages = []
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as log:
            for row in csv.DictReader(log):
                begin = datetime.fromisoformat(row['start_utc']).timestamp()
                finish = datetime.fromisoformat(row['end_utc']).timestamp()
                if begin >= start and finish <= end:
                    target = float(row['target_bpm']) if row['target_bpm'] else np.nan
                    stages.append((begin, finish, row['kind'], target, row['label']))
    return sorted(stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='session folder written by tools/save_jpegs.py')
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--pose-every', type=int, default=3, help='run the pose model on every Nth frame')
    parser.add_argument('--overlay', type=int, default=0, help='save every Nth pose frame with the skeleton drawn on')
    parser.add_argument('--pace', type=Path, nargs='*',
                        help='paced-breathing logs; default: any in output/ that overlap the session')
    args = parser.parse_args()

    session = load_session(args.session)
    first = next((f for f in map(read_frame, session.files) if f is not None), None)
    if first is None:
        raise SystemExit('No readable frames in this session.')
    size = first.shape[:2]
    calibration = load_calibration(args.calibration, (size[1], size[0]))
    box = tuple(calibration['box'])
    tracker = Tracker(PoseModel(), (size[1], size[0]), calibration['head_deg'])
    motion = Motion(box)
    out = OUTPUT / session.name
    out.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        (out / 'overlay').mkdir(exist_ok=True)

    region, region_id = box, 0
    times, rows, signals, reference, regions = [], [], [], [], []
    pose_times, pose_rows, labels, pose_scores = [], [], [], []
    skipped = 0
    for index, (path, time) in enumerate(zip(session.files, session.time)):
        frame = read_frame(path, size)
        if frame is None or (times and time <= times[-1]):
            skipped += 1
            continue
        if index % args.pose_every == 0:
            points, scores = tracker(frame)
            torso = torso_box(points, scores, box)
            if torso and overlap(torso, region) < REGION_KEEP:
                region, region_id = torso, region_id + 1
            width, face = position.features(points, scores)
            pose_times.append(time)
            labels.append(position.classify(width, face))
            pose_scores.append(float(scores[:UPPER].mean()))
            pose_rows.append([session.sequence[index], utc(time), f'{width:.3f}', f'{face:.3f}', labels[-1]]
                             + [f'{v:.1f}' for p, s in zip(points, scores) for v in (*p, s * 100)])
            if args.overlay and (len(pose_times) - 1) % args.overlay == 0:
                image = draw(frame, points, scores, [(box, (160, 160, 160)), (region, (0, 200, 0))])
                cv2.imwrite(str(out / 'overlay' / f'{session.sequence[index]:010d}.jpg'), image)
        moved, moving, torso_moving, signal, room = motion.step(frame, time, region)
        times.append(time)
        rows.append((moved, moving, torso_moving))
        signals.append(signal)
        reference.append(room)
        regions.append(region_id)
        if len(times) % 1000 == 0:
            print(f'{len(times)} / {len(session.files)} frames')
    if len(times) < 2:
        raise SystemExit('Too few readable frames to analyse.')

    times = np.array(times)
    moved, moving, torso_moving = np.array(rows).T
    active = moving >= ACTIVE_FRACTION
    runs = breathing.analyse(times, np.array(signals), np.array(reference), torso_moving, regions)
    pause_list = [p for run in runs for p in breathing.pauses(run)]
    stretches = position.timeline(pose_times, labels)
    rolls = position.roll_overs(stretches, times, active)

    # Epochs are aligned to :00 and :30 of UTC so they line up with other devices.
    step = np.minimum(np.diff(times, prepend=times[0]), MAX_GAP_S)
    step[np.diff(times, prepend=times[0]) > MAX_GAP_S] = 0
    pose_times_array, pose_scores = np.array(pose_times), np.array(pose_scores)
    epochs = []
    for start in np.arange(np.floor(times[0] / EPOCH_S) * EPOCH_S, times[-1], EPOCH_S):
        inside = (times >= start) & (times < start + EPOCH_S)
        if not inside.any():
            continue
        centre = start + EPOCH_S / 2
        valid = inside & ~np.isnan(moved)
        posed = (pose_times_array >= start) & (pose_times_array < start + EPOCH_S)
        rate, quality = breathing.rate_between(runs, centre - breathing.WINDOW_S / 2, centre + breathing.WINDOW_S / 2)
        epochs.append({
            'start_utc': utc(start),
            'time': centre,
            'coverage': min(1.0, step[inside].sum() / EPOCH_S),
            'motion_mean': moved[valid].mean() if valid.any() else np.nan,
            'motion_max': moved[valid].max() if valid.any() else np.nan,
            'active_fraction': active[valid].mean() if valid.any() else np.nan,
            'position': next((p for a, b, p in stretches if a <= centre <= b), ''),
            'keypoint_score': pose_scores[posed].mean() if posed.any() else np.nan,
            'breaths_per_min': rate,
            'breathing_quality': quality,
            'pauses': sum(start <= a < start + EPOCH_S for a, _ in pause_list),
        })

    pace_paths = args.pace if args.pace is not None else sorted(OUTPUT.glob('pace_*.csv'))
    paced = []
    for begin, finish, kind, target, label in load_pace(pace_paths, times[0], times[-1]):
        if kind == 'hold':
            held = sum(min(b, finish + 5) - max(a, begin - 5) for a, b in pause_list
                       if b > begin - 5 and a < finish + 5)
            paced.append((begin, finish, kind, label, finish - begin, held, np.nan))
        elif kind == 'paced':
            rate, quality = breathing.rate_between(runs, begin + PACE_SKIP_S, finish)
            paced.append((begin, finish, kind, label, target, rate, quality))

    def write(name, header, lines):
        with (out / name).open('w', newline='', encoding='utf-8') as log:
            csv.writer(log).writerows([header, *lines])

    write('pose.csv', ['sequence', 'time_utc', 'shoulder_width', 'nose_offset', 'position']
          + [f'{n}_{part}' for n in NAMES for part in ('x', 'y', 'score')], pose_rows)
    write('frames.csv', ['time_utc', 'motion_px_s', 'moving_fraction', 'torso_moving_fraction', 'region'],
          [[utc(t), f'{a:.3f}', f'{b:.4f}', f'{c:.4f}', r] for t, (a, b, c), r in zip(times, rows, regions)])
    write('epochs.csv', [k for k in epochs[0] if k != 'time'],
          [[f'{v:.3f}' if isinstance(v, float) else v for k, v in e.items() if k != 'time'] for e in epochs])
    events = [(t, 'roll_over', f'{a} to {b}', '') for t, a, b in rolls]
    events += [(a, 'breathing_pause', '', f'{b - a:.1f}') for a, b in pause_list]
    write('events.csv', ['time_utc', 'event', 'detail', 'duration_s'],
          [[utc(t), *rest] for t, *rest in sorted(events)])
    if paced:
        write('paced.csv', ['start_utc', 'end_utc', 'kind', 'label', 'target', 'measured', 'quality'],
              [[utc(a), utc(b), kind, label, f'{x:.1f}', f'{y:.1f}', f'{q:.2f}'] for a, b, kind, label, x, y, q in paced])
    report.summary(out / 'summary.png', session.name, times, moved, epochs, runs, pause_list,
                   stretches, rolls, paced)

    rates = [e['breaths_per_min'] for e in epochs if e['breathing_quality'] >= breathing.GOOD_QUALITY]
    print(f'\n{session.name}: {len(times)} frames over {(times[-1] - times[0]) / 60:.1f} min'
          + (f', {skipped} unreadable or out of order' if skipped else ''))
    print(f'Breathing: {np.median(rates):.1f} breaths/min median over {len(rates)} good epochs'
          if rates else 'Breathing: no epochs with a clear signal')
    print(f'Roll-overs: {len(rolls)}. Breathing pauses of {breathing.PAUSE_MIN_S:.0f} s or more: {len(pause_list)} '
          '(a flag for interest, not a medical measurement).')
    for begin, finish, kind, label, target, measured, quality in paced:
        if kind == 'hold':
            print(f'  {label or "hold"}: held {target:.0f} s, pause detected for {measured:.0f} s')
        else:
            print(f'  {label or "paced"}: target {target:.0f}, measured {measured:.1f} breaths/min (quality {quality:.2f})')
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
