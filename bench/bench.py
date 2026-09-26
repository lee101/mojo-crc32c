"""Correctness-gated benchmark for mojo-crc32c.

Every case checks the checksum against the real `crc32c` extension (or zlib)
before timing, so a wrong kernel shows up as a correctness failure rather than
a suspiciously good number.

The honest baseline here is not NumPy: a CRC is a sequential byte fold with no
vectorised formulation, so the comparisons are the two things a caller would
actually reach for, namely the upstream SSE4.2/ARMv8 CRC instruction and zlib's
own CRC-32.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_crc32c as mcrc  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)
import crc32c as real_crc32c  # noqa: E402
import zlib  # noqa: E402

SIZE = 64 << 20
REPEATS = 5


def _time(fn, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _report(label, ref, got, size=SIZE):
    ratio = ref / got if got else float("nan")
    gbs_ref = size / ref / 1e9
    gbs_got = size / got / 1e9
    print(
        f"{label:<26}{ref*1e3:>9.1f}ms{gbs_ref:>8.2f}{got*1e3:>9.1f}ms"
        f"{gbs_got:>8.2f}{ratio:>8.2f}x"
    )


def main():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, SIZE, dtype=np.uint8).tobytes()

    truth_c = real_crc32c.crc32c(data)
    truth_i = zlib.crc32(data)
    assert mcrc.crc32c(data) == truth_c
    assert mcrc.crc32(data) == truth_i
    for workers in (2, 4, 8, 16):
        assert mcrc.crc32c_threaded(data, 0, workers) == truth_c, workers

    print(
        f"{'case':<26}{'reference':>10}{'GB/s':>8}{'mojo-crc32c':>10}"
        f"{'GB/s':>8}{'ratio':>9}   (vs)"
    )
    print("-" * 78)

    for width in (8, 16):
        fn = lambda w=width: mcrc.crc32c_width(data, 0, w)
        assert fn() == truth_c
        _report(
            f"crc32c slice-by-{width}",
            _time(lambda: real_crc32c.crc32c(data)),
            _time(fn),
        )

    for workers in (2, 4, 8, 16):
        fn = lambda k=workers: mcrc.crc32c_threaded(data, 0, k)
        _report(
            f"crc32c threaded x{workers}",
            _time(lambda: real_crc32c.crc32c(data)),
            _time(fn),
        )

    _report("crc32 (IEEE)", _time(lambda: zlib.crc32(data)), _time(lambda: mcrc.crc32(data)))

    # A 1 MiB buffer, where the per-call thread hand-off is a visible cost.
    small = data[: 1 << 20]
    truth_small = real_crc32c.crc32c(small)
    assert mcrc.crc32c_threaded(small, 0, 8) == truth_small
    print()
    _report(
        "crc32c 1 MiB (threaded x8)",
        _time(lambda: real_crc32c.crc32c(small), 50),
        _time(lambda: mcrc.crc32c_threaded(small, 0, 8), 50),
        size=1 << 20,
    )
    _report(
        "crc32c 1 MiB (serial x16)",
        _time(lambda: real_crc32c.crc32c(small), 50),
        _time(lambda: mcrc.crc32c_width(small, 0, 16), 50),
        size=1 << 20,
    )


if __name__ == "__main__":
    main()
