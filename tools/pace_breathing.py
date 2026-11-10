"""Paced-breathing test: beeps a known breathing rate while the camera records.

Start tools/save_jpegs.py first, then run this and lie down. Breathe in on the
high beep and out on the low beep. Three quick beeps mean "hold your breath";
one long beep means "breathe again". Standard library only (Windows sound).
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import winsound

HIGH, LOW = 880, 440


def beep(frequency, ms=120):
    winsound.Beep(frequency, ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='', help='position being tested, e.g. back, left, right, front')
    parser.add_argument('--rates', type=float, nargs='+', default=[8, 12, 18, 24], help='breaths per minute')
    parser.add_argument('--stage', type=float, default=60, help='seconds at each rate')
    parser.add_argument('--hold', type=float, default=20, help='breath-hold seconds; 0 skips it')
    parser.add_argument('--lead', type=float, default=20, help='seconds to get into position first')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_path = args.output / f'pace_{started:%Y%m%dT%H%M%S}.csv'
    total = args.lead + len(args.rates) * args.stage + (60 + args.hold if args.hold else 0)
    print(f'About {total / 60:.1f} minutes. High beep: breathe in. Low beep: breathe out.')
    print(f'You have {args.lead:.0f} s to lie down. Ctrl+C stops.')

    def stage(kind, target, seconds, action):
        begin = datetime.now(timezone.utc)
        action(time.monotonic() + seconds)
        writer.writerow([args.label, kind, target, begin.isoformat(), datetime.now(timezone.utc).isoformat()])
        log.flush()

    def wait(until):
        time.sleep(max(0, until - time.monotonic()))

    def paced(rate):
        def run(until):
            period = 60 / rate
            beat = time.monotonic()
            while beat < until:
                beep(HIGH)
                wait(beat + period * 0.4)   # breathing in takes a little less time than out
                beep(LOW)
                beat += period
                wait(beat)
        return run

    with log_path.open('x', newline='', encoding='utf-8') as log:
        writer = csv.writer(log)
        writer.writerow(['label', 'kind', 'target_bpm', 'start_utc', 'end_utc'])
        try:
            wait(time.monotonic() + args.lead)
            for rate in args.rates:
                print(f'{rate:g} breaths per minute')
                stage('paced', rate, args.stage, paced(rate))
            if args.hold:
                beep(LOW, 600)
                print('Breathe normally')
                stage('free', '', 30, wait)
                for _ in range(3):
                    beep(HIGH, 80)
                print(f'Hold your breath for {args.hold:.0f} s')
                stage('hold', '', args.hold, wait)
                beep(LOW, 600)
                print('Breathe normally')
                stage('free', '', 30, wait)
            beep(LOW, 600)
        except KeyboardInterrupt:
            pass
    print(f'Saved {log_path}')


if __name__ == '__main__':
    main()
