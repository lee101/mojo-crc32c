"""mojo-crc32c: CRC-32C and IEEE CRC-32 with Mojo slice-by-N folding kernels.

Installable alongside the real `crc32c` package, which the parity tests
compare against byte for byte.
"""

from .core import (
    CRC32CHash,
    combine,
    crc32,
    crc32c,
    crc32c_threaded,
    crc32c_width,
)

__all__ = [
    "CRC32CHash",
    "combine",
    "crc32",
    "crc32c",
    "crc32c_threaded",
    "crc32c_width",
]
__version__ = "0.1.0"
