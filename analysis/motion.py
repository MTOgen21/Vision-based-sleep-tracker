"""Per-frame movement and raw breathing signals from optical flow."""
import cv2
import numpy as np

from .frames import MAX_GAP_S

FLOW_WIDTH = 320       # the upper-body box is shrunk to this width before flow
GRID = 4               # the torso is split into GRID x GRID cells
# Breathing moves the image well under 2 px/s; anything this fast is real movement.
# Starting values, in full-resolution pixels: tune on real recordings.
MOVING_PX_S = 6.0
ACTIVE_FRACTION = 0.01  # a frame is 'active' when this share of the box is moving


class Motion:
    """Feed frames in order; each step returns movement and the torso's raw signals."""

    def __init__(self, box):
        self.x1, self.y1, self.x2, self.y2 = (int(round(c)) for c in box)
        self.shrink = min(1.0, FLOW_WIDTH / (self.x2 - self.x1))
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.previous = None
        self.previous_time = None

    def step(self, frame, time, torso):
        """Return (motion px/s, moving fraction, torso moving fraction, signals, reference).

        `signals` holds, for each torso cell, mean brightness then mean x and y
        velocity (px/s). `reference` is the brightness of everything outside the
        torso, used later to cancel room-light changes. Flow values are NaN on the
        first frame and after a gap.
        """
        grey = cv2.cvtColor(frame[self.y1:self.y2, self.x1:self.x2], cv2.COLOR_BGR2GRAY)
        if self.shrink < 1:
            grey = cv2.resize(grey, None, fx=self.shrink, fy=self.shrink, interpolation=cv2.INTER_AREA)
        height, width = grey.shape
        # Torso box in the shrunken image, at least one pixel per grid cell.
        tx1 = int(np.clip((torso[0] - self.x1) * self.shrink, 0, width - GRID))
        ty1 = int(np.clip((torso[1] - self.y1) * self.shrink, 0, height - GRID))
        tx2 = int(np.clip((torso[2] - self.x1) * self.shrink, tx1 + GRID, width))
        ty2 = int(np.clip((torso[3] - self.y1) * self.shrink, ty1 + GRID, height))

        def cells(image):
            return cv2.resize(image[ty1:ty2, tx1:tx2], (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()

        brightness = grey.astype(np.float32)
        inside = brightness[ty1:ty2, tx1:tx2]
        outside = brightness.size - inside.size
        reference = (brightness.sum() - inside.sum()) / outside if outside else 0.0
        signals = np.full(3 * GRID * GRID, np.nan, np.float32)
        signals[:GRID * GRID] = cells(brightness)
        motion = moving = torso_moving = np.nan
        if self.previous is not None and 0 < time - self.previous_time <= MAX_GAP_S:
            # Flow is per frame in shrunken pixels; convert to full-resolution px/s.
            velocity = self.flow.calc(self.previous, grey, None) / (self.shrink * (time - self.previous_time))
            speed = np.hypot(velocity[..., 0], velocity[..., 1])
            motion = float(speed.mean())
            moving = float((speed > MOVING_PX_S).mean())
            torso_moving = float((speed[ty1:ty2, tx1:tx2] > MOVING_PX_S).mean())
            signals[GRID * GRID:2 * GRID * GRID] = cells(velocity[..., 0])
            signals[2 * GRID * GRID:] = cells(velocity[..., 1])
        self.previous, self.previous_time = grey, time
        return motion, moving, torso_moving, signals, reference
