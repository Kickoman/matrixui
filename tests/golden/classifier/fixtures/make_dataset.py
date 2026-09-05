#!/usr/bin/env python3
"""Deterministic 8x8 grayscale PNG dataset for the classifier/GAN goldens.

Writes <out>/<label>/img_<i>.png for labels 0..9, two images per label, using
only the standard library. Seeded, so the bytes are identical on every run;
the committed fixtures/network.wgt was trained on exactly this dataset (see
make_fixtures.sh). Regenerating with a different seed or size invalidates the
recorded expectations AND the committed .wgt fixtures.
"""

import os
import random
import struct
import sys
import zlib

SEED = 20260831
SIDE = 8
LABELS = 10
PER_LABEL = 2


def write_png(path, pixels, side):
    def chunk(tag, data):
        payload = tag + data
        return (struct.pack(">I", len(data)) + payload
                + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0)  # 8-bit grayscale
    raw = b"".join(
        b"\x00" + bytes(pixels[y * side:(y + 1) * side]) for y in range(side)
    )
    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n"
                  + chunk(b"IHDR", header)
                  + chunk(b"IDAT", zlib.compress(raw))
                  + chunk(b"IEND", b""))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: make_dataset.py <output-dir>")
    root = sys.argv[1]
    rng = random.Random(SEED)
    for label in range(LABELS):
        directory = os.path.join(root, str(label))
        os.makedirs(directory, exist_ok=True)
        for i in range(PER_LABEL):
            # A label-specific diagonal stripe over mild noise, so the classes
            # are actually separable and a tiny network can learn something.
            pixels = []
            for y in range(SIDE):
                for x in range(SIDE):
                    stripe = 32 if (x + y) % LABELS == label else 224
                    pixels.append(max(0, min(255, stripe + rng.randrange(-24, 25))))
            write_png(os.path.join(directory, f"img_{i}.png"), pixels, SIDE)
    print(f"dataset: {LABELS} labels x {PER_LABEL} images, {SIDE}x{SIDE}, seed {SEED}")


if __name__ == "__main__":
    main()
