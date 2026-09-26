"""Public API of mojo-crc32c, shaped like the upstream `crc32c` package."""

from concurrent.futures import ThreadPoolExecutor

from ._lib import (
    CRC32C_POLY,
    IEEE_POLY,
    combine as _combine,
    crc32 as _crc32,
    crc32c as _crc32c,
    crc32c_width as _crc32c_width,
)

__all__ = [
    "CRC32CHash",
    "combine",
    "crc32",
    "crc32c",
    "crc32c_threaded",
    "crc32c_width",
]


def crc32c(data, value: int = 0) -> int:
    """Calculate crc32c incrementally, like `crc32c.crc32c`."""
    return _crc32c(data, value)


def crc32(data, value: int = 0) -> int:
    """IEEE CRC-32 of `data`; see `mojo_crc32c.crc32`."""
    return _crc32(data, value)


def crc32c_width(data, value: int = 0, width: int = 16) -> int:
    """CRC-32C through a specific slice-by-`width` folding kernel."""
    return _crc32c_width(data, value, width)


def combine(crc1: int, crc2: int, len2: int, poly: int = CRC32C_POLY) -> int:
    """Merge two consecutive CRC-32C checksums."""
    return _combine(crc1, crc2, len2, poly)


def crc32c_threaded(data, value: int = 0, workers: int = 8) -> int:
    """CRC-32C of `data` split across `workers` threads.

    Each worker folds its own slice; the leading one starts from `value` and
    the rest from zero, and the partials are stitched back together with the
    GF(2) combine operator, so the result is bit-identical to
    `crc32c(data, value)`. Measured on the shared host this port was developed
    on, threading never paid: the fold is L1-bound and the workers only
    contend, so the default `crc32c` stays serial. Below 4 MiB the hand-off
    alone costs several times the whole fold, so this falls back to serial.
    """
    mv = memoryview(data)
    if not mv.contiguous:
        raise ValueError("input buffer must be C-contiguous")
    mv = mv.cast("B")
    n = mv.nbytes
    if n == 0:
        return value & 0xFFFFFFFF
    if workers < 2 or n < 1 << 22:
        return _crc32c(mv, value)
    step = -(-n // workers)
    parts = [(off, min(off + step, n)) for off in range(0, n, step)]
    seeds = [value & 0xFFFFFFFF] + [0] * (len(parts) - 1)
    with ThreadPoolExecutor(max_workers=len(parts)) as ex:
        crcs = list(
            ex.map(
                lambda job: _crc32c(mv[job[0][0]:job[0][1]], job[1]),
                zip(parts, seeds),
            )
        )
    acc = crcs[0]
    for part, crc in zip(parts[1:], crcs[1:]):
        acc = _combine(acc, crc, part[1] - part[0], CRC32C_POLY)
    return acc


class CRC32CHash:
    """hashlib-shaped wrapper around `crc32c`, as in upstream `CRC32CHash`."""

    @property
    def digest_size(self) -> int:
        return 4

    @property
    def block_size(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "crc32c"

    @property
    def checksum(self) -> int:
        return self._checksum

    def __init__(self, data=b"", gil_release_mode: int = -1) -> None:
        # `gil_release_mode` is accepted for signature compatibility: ctypes
        # already releases the GIL for the foreign call.
        self._gil_release_mode = gil_release_mode
        self._checksum = _crc32c(data, 0)

    def update(self, data) -> None:
        self._checksum = _crc32c(data, self._checksum)

    def digest(self) -> bytes:
        return self._checksum.to_bytes(4, "big")

    def hexdigest(self) -> str:
        return self.digest().hex()

    def copy(self) -> "CRC32CHash":
        res = type(self)(gil_release_mode=self._gil_release_mode)
        res._checksum = self._checksum
        return res
