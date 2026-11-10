"""RTMPose-m (body7) on onnxruntime: 17 COCO keypoints from one frame."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'rtmpose-m_body7.onnx'
# Must match tools/fetch_model.py. Any other file is refused.
MODEL_SHA256 = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
INPUT_W, INPUT_H = 192, 256
# Preprocessing from the model's own pipeline.json: RGB order, these means and scales.
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
SIMCC_SPLIT = 2.0
MIN_SCORE = 0.3
NAMES = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder',
         'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
         'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12
# Legs are out of frame by design, so only nose to hips is ever used.
UPPER = 13
TORSO = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12)]


def crop_matrix(box, head_deg, padding):
    """2x3 matrix mapping model-input pixels to frame pixels, with the head pointing up.

    head_deg is where the head points in the frame: 0 up, 90 right, 180 down, -90 left.
    """
    x1, y1, x2, y2 = box
    angle = np.deg2rad(head_deg)
    u = np.array([np.cos(angle), np.sin(angle)])    # model x axis, in frame coordinates
    v = np.array([-np.sin(angle), np.cos(angle)])   # model y axis
    w, h = x2 - x1, y2 - y1
    width = (abs(w * u[0]) + abs(h * u[1])) * padding
    height = (abs(w * v[0]) + abs(h * v[1])) * padding
    scale = max(width, height * INPUT_W / INPUT_H) / INPUT_W
    centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    matrix = np.empty((2, 3))
    matrix[:, 0] = scale * u
    matrix[:, 1] = scale * v
    matrix[:, 2] = centre - scale * (INPUT_W / 2 * u + INPUT_H / 2 * v)
    return matrix


class PoseModel:
    def __init__(self, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise SystemExit('Model missing. Run: py tools/fetch_model.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise SystemExit(f'{path} is not the verified model. Delete it and run tools/fetch_model.py')
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        model_input = self.session.get_inputs()[0]
        outputs = {o.name: o.shape for o in self.session.get_outputs()}
        assert model_input.shape[1:] == [3, INPUT_H, INPUT_W], model_input.shape
        assert outputs['simcc_x'][2] == INPUT_W * SIMCC_SPLIT, outputs
        assert outputs['simcc_y'][2] == INPUT_H * SIMCC_SPLIT, outputs
        self.input = model_input.name

    def __call__(self, frame, box, head_deg, padding=1.25):
        """Return (17, 2) keypoints in frame pixels and (17,) scores."""
        matrix = crop_matrix(box, head_deg, padding)
        crop = cv2.warpAffine(frame, matrix, (INPUT_W, INPUT_H),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        tensor = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
        simcc_x, simcc_y = self.session.run(['simcc_x', 'simcc_y'],
                                            {self.input: tensor.transpose(2, 0, 1)[None]})
        model_xy = np.stack([simcc_x[0].argmax(1), simcc_y[0].argmax(1)], 1) / SIMCC_SPLIT
        # Same rule as mmpose: a keypoint is only as certain as its weaker axis.
        scores = np.minimum(simcc_x[0].max(1), simcc_y[0].max(1))
        points = model_xy @ matrix[:, :2].T + matrix[:, 2]
        return points.astype(np.float32), scores.astype(np.float32)


class Tracker:
    """Runs the model on a tight box around the last pose, or the whole frame when lost."""

    def __init__(self, model, frame_size, head_deg):
        self.model = model
        self.full = (0, 0, frame_size[0], frame_size[1])
        self.head_deg = head_deg
        self.box = None

    def __call__(self, frame):
        if self.box is None:
            points, scores = self.model(frame, self.full, self.head_deg, padding=1.0)
        else:
            points, scores = self.model(frame, self.box, self.head_deg)
        seen = scores[:UPPER] >= MIN_SCORE
        if seen.sum() >= 4:
            found = points[:UPPER][seen]
            self.box = (*found.min(0), *found.max(0))
        else:
            self.box = None
        return points, scores


def torso_box(points, scores, limit):
    """Box around shoulders and hips, clipped to `limit`, or None if any is uncertain."""
    if scores[TORSO].min() < MIN_SCORE:
        return None
    x1, y1 = points[TORSO].min(0)
    x2, y2 = points[TORSO].max(0)
    # Lying on the side stacks the shoulders, so keep a sensible minimum width.
    side = 0.6 * max(x2 - x1, y2 - y1)
    half_w = max(x2 - x1, side) * 1.15 / 2
    half_h = max(y2 - y1, side) * 1.15 / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (max(cx - half_w, limit[0]), max(cy - half_h, limit[1]),
           min(cx + half_w, limit[2]), min(cy + half_h, limit[3]))
    return box if box[2] - box[0] >= 16 and box[3] - box[1] >= 16 else None


def overlap(a, b):
    """Intersection over union of two boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)


def draw(frame, points, scores, boxes=()):
    """Draw the upper-body skeleton and any (box, colour) pairs onto a copy of the frame."""
    image = frame.copy()
    for box, colour in boxes:
        x1, y1, x2, y2 = (int(round(c)) for c in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    for a, b in SKELETON:
        if min(scores[a], scores[b]) >= MIN_SCORE:
            cv2.line(image, tuple(map(int, points[a])), tuple(map(int, points[b])), (255, 200, 0), 2)
    for index in range(UPPER):
        if scores[index] >= MIN_SCORE:
            # Subject's left is drawn orange, right blue, to make swaps easy to spot.
            colour = (0, 140, 255) if index % 2 else (255, 120, 40)
            cv2.circle(image, tuple(map(int, points[index])), 4, colour, -1)
    return image
