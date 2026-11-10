"""Check the camera framing and save the upper-body box.

Record about 20 s lying still on your back, then:  py -m analysis.calibrate raw_image/<session>
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from .frames import CALIBRATION, load_session
from .pose import L_HIP, L_SHOULDER, MIN_SCORE, R_HIP, R_SHOULDER, TORSO, PoseModel, draw
from .run import read_frame

SAMPLES = 30
HEAD_ABOVE = 0.6     # top of the head sits this many shoulder widths beyond the face
HIPS_BELOW = 0.25    # the box ends this share of the torso length past the hips
OK_CM = 5.0
FILL = 0.8           # the upper body should span about this share of the frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--shoulder-cm', type=float, default=36.0,
                        help='distance between your shoulder joints, used only to give advice in cm')
    parser.add_argument('--output', type=Path, default=CALIBRATION)
    args = parser.parse_args()

    session = load_session(args.session)
    picks = np.linspace(0, len(session.files) - 1, min(SAMPLES, len(session.files))).astype(int)
    frames = [f for f in (read_frame(session.files[i]) for i in picks) if f is not None]
    if not frames:
        raise SystemExit('No readable frames in this session.')
    height, width = frames[0].shape[:2]
    full = (0, 0, width, height)
    model = PoseModel()

    # The head direction is unknown at first, so try all four and keep the most confident.
    test = frames[len(frames) // 2]
    head_deg = max((0, 90, 180, -90), key=lambda a: model(test, full, a, padding=1.0)[1][TORSO].min())
    poses = []
    for frame in frames:
        points, scores = model(frame, full, head_deg, padding=1.0)
        seen = scores >= MIN_SCORE
        if seen[TORSO].all():
            # Second pass on a tight box around the person for sharper keypoints.
            points, scores = model(frame, (*points[seen].min(0), *points[seen].max(0)), head_deg)
            poses.append((points, scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    picture = args.output.with_suffix('.jpg')
    if len(poses) < len(frames) / 2:
        points, scores = model(test, full, head_deg, padding=1.0)
        cv2.imwrite(str(picture), draw(test, points, scores))
        raise SystemExit(f'Shoulders and hips were found in only {len(poses)} of {len(frames)} frames. '
                         f'Check {picture}: both shoulders and both hips must be in view, uncovered.')
    points = np.median([p for p, _ in poses], axis=0)
    scores = np.median([s for _, s in poses], axis=0)

    shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
    hips = (points[L_HIP] + points[R_HIP]) / 2
    spine = shoulders - hips
    torso = np.hypot(*spine)
    head = spine / torso
    left = points[L_SHOULDER] - points[R_SHOULDER]
    shoulder_px = np.hypot(*left)
    if left @ np.array([-head[1], head[0]]) <= 0:
        print('Warning: this looks face-down or the shoulders were mixed up. Calibrate lying on your back.')
    mm_per_px = args.shoulder_cm * 10 / shoulder_px
    cm = mm_per_px / 10

    # Snap the head direction to the nearest side of the frame.
    head_deg = int(round(np.degrees(np.arctan2(head[0], -head[1])) / 90)) * 90
    head_deg = -90 if head_deg == 270 else 180 if head_deg == -180 else head_deg
    along = 1 if head_deg in (0, 180) else 0          # image axis the body lies along
    face = points[:5][scores[:5] >= MIN_SCORE]
    face_centre = face.mean(0) if len(face) else shoulders + head * shoulder_px * 0.6
    top = (face_centre + head * shoulder_px * HEAD_ABOVE)[along]
    bottom = (hips - head * torso * HIPS_BELOW)[along]
    low, high = sorted((top, bottom))
    extent = (width, height)[along]
    box = [0, 0, width, height]
    box[along], box[along + 2] = max(0, int(low)), min(extent, int(np.ceil(high)))

    # Where the upper body sits relative to the frame centre, in the subject's own terms.
    offset = np.array([width / 2, height / 2])
    offset[along] = (low + high) / 2
    offset[1 - along] = shoulders[1 - along]
    offset -= np.array([width / 2, height / 2])
    to_head = offset @ head * cm
    to_left = offset @ (left / shoulder_px) * cm
    advice = []
    if extent == max(width, height) and width != height:
        advice.append('Rotate the camera 90 degrees so the long side of the image runs across the bed.')
    if abs(to_head) > OK_CM:
        advice.append(f'Move the camera about {abs(to_head):.0f} cm towards your {"head" if to_head > 0 else "feet"}.')
    if abs(to_left) > OK_CM:
        advice.append(f'Move the camera about {abs(to_left):.0f} cm towards your {"left" if to_left > 0 else "right"} side.')
    fill = (high - low) / extent
    if fill > 0.97:
        advice.append(f'Your upper body does not fit. Raise the camera by about {fill / FILL - 1:.0%} of its height.')
    elif fill < 0.6:
        advice.append(f'Optional: lowering the camera by about {1 - fill / FILL:.0%} of its height gives more detail.')

    args.output.write_text(json.dumps({
        'box': box, 'head_deg': head_deg, 'mm_per_px': round(float(mm_per_px), 3),
        'frame_size': [width, height], 'session': session.name,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }, indent=2) + '\n', encoding='utf-8')
    cv2.imwrite(str(picture), draw(frames[len(frames) // 2], points, scores, [(box, (0, 200, 0))]))

    print(f'Pose found in {len(poses)} of {len(frames)} frames. About {mm_per_px:.1f} mm per pixel.')
    print(f'Upper-body box {box}, head pointing {({0: "up", 90: "right", 180: "down", -90: "left"})[head_deg]} in the image.')
    print('\n'.join(advice) if advice else 'Framing is good.')
    if advice:
        print('Adjust, record again and re-run until this says the framing is good.')
    print(f'Saved {args.output} and {picture} (check the picture: dots should sit on your joints).')


if __name__ == '__main__':
    main()
