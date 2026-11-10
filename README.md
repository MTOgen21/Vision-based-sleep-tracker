# Waker

An overhead ESP32 camera that measures breathing rate, movement and lying position, as a step towards camera-based sleep tracking.

```
ESP32-CAM  --private Wi-Fi, AES-256-GCM-->  ESP32 DevKit  --USB serial-->  PC  -->  offline analysis
```

Neither board connects to any other network or the internet. This is a proof of concept for a lit room with the upper body uncovered. It is not a medical device.

## Hardware

- **Camera:** AI Thinker ESP32-CAM with PSRAM, mounted above the bed looking down at the upper body. The pin map in `camera.cpp` is for this board.
- **Bridge:** any ESP32 DevKit (esp32dev).

## Setup

1. Install the **PlatformIO IDE** extension in VS Code and open the `firmware` folder.
2. Generate keys (once), from the repository root:

   ```
   py tools/setup_keys.py
   ```

   This creates `firmware/include/secrets.h` with a random Wi-Fi name, password and AES key. It is git-ignored. Don't share it, or `firmware/.pio/` (compiled firmware contains the key). If either is exposed, delete `secrets.h`, run the script again and reflash both boards.

3. Flash both boards from the `firmware` folder, using your actual COM ports:

   ```
   pio run -e camera -t upload --upload-port COM6
   pio run -e bridge -t upload --upload-port COM7
   ```

4. Create the Python environment and fetch the pose model:

   ```
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install --only-binary :all: --require-hashes -r requirements.txt
   .venv\Scripts\python tools/fetch_model.py
   ```

   `requirements.txt` pins every package by SHA-256 (Windows, Python 3.12). The model is downloaded from OpenMMLab over HTTPS and refused unless it matches a fixed checksum.

## Use

Run everything from the repository root with `.venv\Scripts\python`.

1. **Record.** Room lights in their final state, then:

   ```
   .venv\Scripts\python tools/save_jpegs.py --port COM7
   ```

   Each run saves JPEGs and `frames.csv` to a new folder in `raw_image/`.

2. **Calibrate the framing** (once per camera position). Record about 20 s lying still on your back, then:

   ```
   .venv\Scripts\python -m analysis.calibrate raw_image/<session>
   ```

   It tells you how far to move the camera, and saves the upper-body box to `output/calibration.json`. Repeat until it says the framing is good.

3. **Paced-breathing test** (optional, checks accuracy). While recording, in a second terminal:

   ```
   .venv\Scripts\python tools/pace_breathing.py --label back
   ```

   It beeps set breathing rates and a breath-hold, and logs what it asked for.

4. **Analyse:**

   ```
   .venv\Scripts\python -m analysis.run raw_image/<session> --overlay 20
   ```

   Results go to `output/<session>/`: `summary.png` (movement, breathing rate and depth, position, data quality), per-frame, per-epoch and event CSVs, and overlay images for checking the pose. Paced tests that overlap the session are compared automatically.

Tests use a synthetic recording with known breathing: `.venv\Scripts\python -m unittest tests.test_analysis` (takes a couple of minutes).

## How it works

- **Movement:** dense optical flow inside the upper-body box, in pixels per second.
- **Breathing:** while the body is still, brightness and sub-pixel motion in a grid over the torso are band-passed to 8 to 39 breaths/min and combined into one waveform. Rate comes from its spectrum; a pause is flagged when its depth collapses for 10 s or more.
- **Position:** RTMPose-m keypoints give back, left side, right side or front. A roll-over is a change of position that coincides with movement.
- **Camera settings:** 6 FPS, 50 Hz anti-flicker, and exposure locked a few seconds after each connection so brightness changes come from the scene only. Reset the bridge to re-lock after the lighting changes.

## Limits

- Lit room and uncovered upper body only. Darkness needs an infrared camera and lamp.
- Thresholds are starting values, checked on synthetic data and one real recording, and still need calibrating (see [Status](#status)).
- Face-down is the least reliable position.
- Breathing pauses are a flag for interest, not a diagnosis.
- Future: real-time analysis, running the same per-frame pipeline live alongside `save_jpegs.py`.

## Performance

640x480 JPEG at quality 15, up to 6 FPS. Frames over 64 KiB are dropped. USB serial runs at 921600 baud (about 90 KB/s), so 6 FPS needs JPEGs averaging under about 15 KB; larger frames lower the rate, which the analysis tolerates because it works from timestamps.

## Security

- **Protected:** each JPEG is encrypted and authenticated with AES-256-GCM. Without the key, anyone nearby can neither view images nor inject fake or replayed ones. The Wi-Fi link also uses WPA2.
- **Not protected:** someone nearby can jam the link or drop and delay frames. Keys are stored unencrypted in each board's flash (no secure boot or flash encryption), so physical access exposes them. Images are plaintext over the USB cable and on the PC, in `raw_image/` and `output/`.

## Protocol (version 2)

When the bridge connects (TCP 192.168.50.1:7443), it sends the camera a random 16-byte challenge. Each frame is then:

| Field | Bytes |
|---|---|
| `SLPC` magic | 4 |
| Version (2) | 1 |
| Sequence | 4 |
| Camera uptime, µs | 8 |
| JPEG length | 4 |
| GCM nonce (8-byte random boot ID + sequence) | 12 |
| Encrypted JPEG | length |
| GCM tag | 16 |

Integers are big-endian. The header and the connection's challenge are authenticated, so frames recorded from an earlier connection fail. Within a connection the bridge requires the same boot ID and an increasing sequence.

The bridge's USB output is the first 21 header bytes followed by the plaintext JPEG. Timestamps are camera uptime, not wall-clock time.

## Status

Work in progress. The encrypted link, the firmware and the analysis all run on hardware, and the analysis passes its synthetic tests.

### First paced-breathing test (10 Nov 2026)

One person lying on their back, lit room, 5.9 minutes, one minute at each paced rate:

| Target (breaths/min) | Measured | Signal quality (0 to 1) |
|---|---|---|
| 8 | 8.1 | 0.96 |
| 12 | 11.9 | 0.98 |
| 18 | 18.1 | 0.86 |
| 24 | 24.0 | 0.94 |

A 20 s breath-hold was flagged as a pause. How accurately pauses are timed has not been measured yet.

### Not yet calibrated

This is a single favourable recording, not a validation:

- The breathing was deliberately deep, which gives a much stronger signal than the shallow breathing of sleep.
- Only one position, one person and one lighting setup have been tried. The position was classified correctly; the others and roll-over detection are unchecked.
- Frames arrived at about 2 per second, not the 6 targeted, because the brightly lit room made JPEGs of about 25 KB (see [Performance](#performance)). 53 of 774 were unreadable or out of order.
- Thresholds are still the starting values.

Next: shallow breathing, the other positions, pause timing and a full-night recording.

## License

MIT, see [LICENSE](LICENSE).
