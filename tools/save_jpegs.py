"""Receive Waker's 21-byte USB headers and save JPEGs. Requires pyserial."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import struct
import time
import uuid

HEADER = struct.Struct('>4sBIQI')
MAX_JPEG = 64 * 1024
PROTOCOL_VERSION = 2  # must match PROTOCOL_VERSION in firmware/include/link.h


def pop_frame(buffer):
    """Resynchronise after boot logs, corrupt headers or partial serial reads."""
    while True:
        start = buffer.find(b'SLPC')
        if start < 0:
            if len(buffer) > 3:
                del buffer[:-3]
            return None
        if start:
            del buffer[:start]
        if len(buffer) < HEADER.size:
            return None
        _, version, sequence, uptime_us, length = HEADER.unpack_from(buffer)
        if version != PROTOCOL_VERSION or not 4 <= length <= MAX_JPEG:
            del buffer[0]
            continue
        end = HEADER.size + length
        if len(buffer) < end:
            return None
        jpeg = bytes(buffer[HEADER.size:end])
        if not (jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')):
            del buffer[0]
            continue
        del buffer[:end]
        return sequence, uptime_us, jpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, help='DevKit USB port, e.g. COM6')
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'raw_image')
    parser.add_argument('--interval', type=float, default=0,
                        help='Minimum seconds between saves; 0 saves every received frame')
    parser.add_argument('--duration', type=float, default=0,
                        help='Stop after this many seconds; 0 runs until Ctrl+C')
    args = parser.parse_args()
    if args.interval < 0 or args.duration < 0:
        parser.error('interval and duration must be non-negative')
    try:
        import serial
    except ImportError:
        parser.exit(1, 'Install dependency: py -m pip install pyserial\n')
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    # One folder per session keeps a night's frames together and easy to delete.
    folder = args.output / session
    folder.mkdir(parents=True)
    metadata = folder / 'frames.csv'
    buffer = bytearray()
    started = time.monotonic()
    last_save = float('-inf')
    last_bytes = started
    warned = False
    saved = 0
    previous = None
    missing = restarts = 0
    report_at, report_frames, report_bytes = started, 0, 0
    port = None
    try:
        # Avoid requesting a reset through DTR/RTS when opening the DevKit.
        port = serial.Serial(port=None, baudrate=args.baud, timeout=0.25)
        port.dtr = False
        port.rts = False
        port.port = args.port
        port.open()
        with metadata.open('x', newline='', encoding='utf-8') as log:
            writer = csv.writer(log)
            writer.writerow(['filename', 'received_utc', 'received_monotonic_ns',
                             'camera_uptime_us', 'sequence', 'jpeg_bytes'])
            print(f'Listening on {args.port}; saving to {folder.resolve()}')
            print('Ctrl+C stops. Filename time is PC receipt time, not camera exposure time.')
            while not args.duration or time.monotonic() - started < args.duration:
                data = port.read(min(max(port.in_waiting, 1), 65536))
                now = time.monotonic()
                if data:
                    # Discard stale incomplete packets after a long interruption.
                    if now - last_bytes > 5:
                        buffer.clear()
                    last_bytes = now
                    buffer.extend(data)
                if not saved and now - started > 10 and not warned:
                    print('No complete JPEG yet. Check camera power, matching keys and USB port.')
                    warned = True
                while True:
                    frame = pop_frame(buffer)
                    if frame is None:
                        break
                    sequence, uptime_us, jpeg = frame
                    received = datetime.now(timezone.utc)
                    monotonic_ns = time.monotonic_ns()
                    # The camera numbers every frame it sends, so a jump means frames
                    # were lost on the way here; a step back means the camera rebooted.
                    if previous is not None:
                        if sequence > previous + 1:
                            missing += sequence - previous - 1
                        elif sequence <= previous:
                            restarts += 1
                    previous = sequence
                    report_frames += 1
                    report_bytes += len(jpeg)
                    if time.monotonic() - last_save < args.interval:
                        continue
                    filename = f'{received:%Y%m%dT%H%M%S_%fZ}_{session[-8:]}_{sequence:010d}.jpg'
                    target = folder / filename
                    # Write to a temporary path so viewers never see a partial JPEG.
                    temporary = target.with_suffix('.jpg.part')
                    temporary.write_bytes(jpeg)
                    temporary.replace(target)
                    writer.writerow([filename, received.isoformat(), monotonic_ns,
                                     uptime_us, sequence, len(jpeg)])
                    log.flush()
                    last_save = time.monotonic()
                    saved += 1
                if report_frames and now - report_at >= 5:
                    print(f'Saved {saved} frames; {report_frames / (now - report_at):.1f} FPS; '
                          f'average {report_bytes / report_frames / 1000:.1f} KB; '
                          f'{missing} missing')
                    report_at, report_frames, report_bytes = now, 0, 0
    except KeyboardInterrupt:
        pass
    except (serial.SerialException, OSError) as exc:
        print(f'Error: {exc}')
        raise SystemExit(1) from exc
    finally:
        if port is not None:
            port.close()
        print(f'Stopped: {saved} JPEGs saved, {missing} missing, {restarts} camera restarts. '
              f'Session: {folder}')


if __name__ == '__main__':
    main()
