from pathlib import Path
import secrets
p = Path(__file__).resolve().parents[1] / 'firmware/include/secrets.h'
if p.exists():
    raise SystemExit('Keys already exist. Delete secrets.h deliberately to regenerate, then reflash BOTH boards.')
p.write_text('#pragma once\n#define CAM_SSID "Waker-' + secrets.token_hex(3) + '"\n#define CAM_PASSWORD "' + secrets.token_urlsafe(24) + '"\nstatic const unsigned char FRAME_KEY[32] = {' + ','.join('0x'+f'{b:02x}' for b in secrets.token_bytes(32)) + '};\n')
print('Created local keys. Build and flash both boards; do not share secrets.h.')
