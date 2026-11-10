"""End-to-end check on a synthetic recording with known breathing: py -m unittest

A textured "torso" swells and dims by a fraction of a pixel at set rates, stops
for a breath-hold, and shifts once. No real images are needed or stored.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from analysis import position, run
from analysis.breathing import GOOD_QUALITY
from analysis.pose import crop_matrix

FPS = 6
WIDTH, HEIGHT = 640, 480
START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp()
# (seconds, breaths per minute or None for a breath-hold)
PHASES = [(60, 12), (60, 18), (30, 15), (20, None), (40, 15), (60, 24)]
SHIFT_AT = 210.0    # the torso slides sideways here, as in a change of position


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def make_session(folder, depth=1.0):
    rng = np.random.default_rng(1)

    def texture(shape):
        noise = cv2.GaussianBlur(rng.uniform(0, 1, shape).astype(np.float32), (0, 0), 3)
        return (noise - noise.mean()) / noise.std() * 30 + 120

    background, torso = texture((HEIGHT, WIDTH)), texture((260, 200))
    edges = np.cumsum([0] + [seconds for seconds, _ in PHASES])
    rows = []
    phase = 0.0
    previous = 0.0
    for index in range(int(edges[-1] * FPS)):
        t = index / FPS + rng.uniform(-0.02, 0.02)
        rate = PHASES[min(np.searchsorted(edges, t, side='right') - 1, len(PHASES) - 1)][1]
        phase += 2 * np.pi * (rate or 0) / 60 * (t - previous)
        previous = t
        breath = np.sin(phase)
        zoom = 1 + 0.006 * depth * breath
        centre = (torso.shape[1] / 2, torso.shape[0] / 2)
        patch = cv2.warpAffine(torso, cv2.getRotationMatrix2D(centre, 0, zoom), torso.shape[::-1],
                               borderMode=cv2.BORDER_REFLECT) * (1 + 0.01 * depth * breath)
        left = 220 + int(40 * np.clip((t - SHIFT_AT) / 2, 0, 1))
        frame = background.copy()
        frame[140:400, left:left + 200] = patch
        frame += rng.normal(0, 2, frame.shape)
        name = f'{index:06d}.jpg'
        image = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        rows.append([name, iso(START + t + 0.2), 0, int((5 + t) * 1e6), index, 0])
    with (folder / 'frames.csv').open('w', newline='', encoding='utf-8') as log:
        csv.writer(log).writerows([['filename', 'received_utc', 'received_monotonic_ns',
                                    'camera_uptime_us', 'sequence', 'jpeg_bytes'], *rows])
    return edges


class CropTest(unittest.TestCase):
    def test_head_direction_maps_model_up_to_the_head(self):
        box = (100, 50, 500, 450)
        expected = {0: (0, -1), 90: (1, 0), 180: (0, 1), -90: (-1, 0)}
        for head_deg, direction in expected.items():
            matrix = crop_matrix(box, head_deg, 1.25)
            centre = matrix @ [96, 128, 1]
            top = matrix @ [96, 0, 1]
            np.testing.assert_allclose(centre, [300, 250], atol=1e-6)
            np.testing.assert_allclose((top - centre) / np.hypot(*(top - centre)), direction, atol=1e-6)
            # A proper rotation, never a mirror image, or left and right would swap.
            self.assertGreater(np.linalg.det(matrix[:, :2]), 0)


class PositionTest(unittest.TestCase):
    @staticmethod
    def pose(left_shoulder_x, right_shoulder_x, nose_x):
        """Keypoints for someone lying with their head towards the top of the image."""
        points = np.zeros((17, 2), np.float32)
        points[0] = nose_x, 60
        points[5], points[6] = (left_shoulder_x, 100), (right_shoulder_x, 100)
        points[11], points[12] = (300 + (left_shoulder_x - 300) * 0.7, 300), (300 + (right_shoulder_x - 300) * 0.7, 300)
        return points, np.ones(17, np.float32)

    def test_positions(self):
        # Seen from above, someone on their back has their left shoulder on the image's right.
        cases = {'back': (370, 230, 300), 'front': (230, 370, 300),
                 'left': (310, 295, 345), 'right': (295, 310, 255)}
        for expected, arguments in cases.items():
            self.assertEqual(position.classify(*position.features(*self.pose(*arguments))), expected)
        points, scores = self.pose(370, 230, 300)
        scores[11] = 0.1
        self.assertEqual(position.classify(*position.features(points, scores)), position.UNKNOWN)

    def test_roll_over_needs_movement(self):
        times = np.arange(0, 120, 0.5)
        labels = ['back'] * 120 + ['left'] * 120
        labels[30] = 'front'    # one stray frame must not create a stretch
        stretches = position.timeline(times, labels)
        self.assertEqual([s[2] for s in stretches], ['back', 'left'])
        still = np.zeros(len(times), bool)
        self.assertEqual(position.roll_overs(stretches, times, still), [])
        moved = still.copy()
        moved[118:122] = True
        self.assertEqual([(a, b) for _, a, b in position.roll_overs(stretches, times, moved)], [('back', 'left')])


class PipelineTest(unittest.TestCase):
    def analyse(self, root, depth):
        """Record a synthetic session under `root`, analyse it and return the results as rows."""
        session = root / 'raw' / 'synthetic'
        session.mkdir(parents=True)
        edges = make_session(session, depth)
        calibration = root / 'calibration.json'
        calibration.write_text(json.dumps({'box': [0, 40, WIDTH, 440], 'head_deg': 0,
                                           'frame_size': [WIDTH, HEIGHT]}))
        pace = root / 'pace.csv'
        with pace.open('w', newline='') as log:
            writer = csv.writer(log)
            writer.writerow(['label', 'kind', 'target_bpm', 'start_utc', 'end_utc'])
            for (seconds, rate), begin in zip(PHASES, edges):
                kind = 'paced' if rate and seconds >= 60 else 'free' if rate else 'hold'
                writer.writerow(['test', kind, rate if kind == 'paced' else '',
                                 iso(START + begin + 0.5), iso(START + begin + seconds)])
        arguments = ['run', str(session), '--calibration', str(calibration), '--pace', str(pace)]
        with mock.patch.object(run, 'OUTPUT', root / 'out'), mock.patch.object(sys, 'argv', arguments):
            run.main()
        out = root / 'out' / 'synthetic'
        self.assertGreater((out / 'summary.png').stat().st_size, 10000)

        def rows(name):
            with (out / name).open() as log:
                return list(csv.DictReader(log))
        return {name: rows(f'{name}.csv') for name in ('paced', 'events', 'frames', 'epochs')}

    def test_known_breathing_is_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.analyse(Path(temporary), depth=0.3)
        paced = [r for r in result['paced'] if r['kind'] == 'paced']
        self.assertEqual(len(paced), 3)
        for row in paced:
            self.assertAlmostEqual(float(row['measured']), float(row['target']), delta=1.0, msg=row)
        hold = next(r for r in result['paced'] if r['kind'] == 'hold')
        self.assertGreater(float(hold['measured']), 12, hold)
        pauses = [e for e in result['events'] if e['event'] == 'breathing_pause']
        self.assertEqual(len(pauses), 1, pauses)
        moving = [datetime.fromisoformat(f['time_utc']).timestamp() - START
                  for f in result['frames'] if float(f['moving_fraction']) >= 0.01]
        self.assertTrue(moving and all(SHIFT_AT - 1 <= t <= SHIFT_AT + 4 for t in moving),
                        f'{len(moving)} moving frames, {moving[:3]}')

    def test_no_breathing_is_not_reported_as_breathing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.analyse(Path(temporary), depth=0)
        quality = [float(e['breathing_quality']) for e in result['epochs']]
        self.assertLess(np.median(quality), GOOD_QUALITY, quality)
        self.assertFalse([e for e in result['events'] if e['event'] == 'breathing_pause'])


if __name__ == '__main__':
    unittest.main()
