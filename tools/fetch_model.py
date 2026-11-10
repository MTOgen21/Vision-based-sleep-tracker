"""Download the RTMPose-m pose model and refuse anything but the verified file.

Standard library only. The download is checked against a fixed SHA-256 before
anything is unpacked, and only the model itself is taken out of the archive.
"""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
       'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip')
ZIP_SHA256 = 'f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741'
# Must match MODEL_SHA256 in analysis/pose.py.
MODEL_SHA256 = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
MAX_BYTES = 80 * 1024 * 1024
MODELS = Path(__file__).resolve().parents[1] / 'models'
TARGET = MODELS / 'rtmpose-m_body7.onnx'


def main():
    if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == MODEL_SHA256:
        print(f'Verified model already present: {TARGET}')
        return
    MODELS.mkdir(exist_ok=True)
    archive = MODELS / 'rtmpose-m.zip'
    digest = hashlib.sha256()
    size = 0
    print('Downloading the model (about 51 MB)...')
    # urllib verifies the server certificate; redirects away from HTTPS are refused below.
    with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as out:
        if not response.geturl().startswith('https://'):
            raise SystemExit('Refused: the download was redirected away from HTTPS.')
        while chunk := response.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                raise SystemExit('Refused: the download is larger than expected.')
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != ZIP_SHA256:
        archive.unlink()
        raise SystemExit('Refused: the download does not match the verified checksum. Nothing was unpacked.')
    with zipfile.ZipFile(archive) as bundle:
        member = next(n for n in bundle.namelist() if n.endswith('/end2end.onnx'))
        model = bundle.read(member)
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise SystemExit('Refused: the model inside the archive does not match the verified checksum.')
    TARGET.write_bytes(model)
    print(f'Verified and saved {TARGET}')


if __name__ == '__main__':
    main()
