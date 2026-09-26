"""ctypes bridge to the compiled Mojo CRC kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-crc32c.so"

#: Reflected polynomial of CRC-32C (Castagnoli), as used by upstream crc32c.
CRC32C_POLY = 0x82F63B78
#: Reflected polynomial of the IEEE CRC-32 that zlib and gzip use.
IEEE_POLY = 0xEDB88320

_UINT32 = ctypes.c_uint32
_ADDRESS = ctypes.c_int64
_U8P = ctypes.POINTER(ctypes.c_uint8)


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.crc32c_build_table.restype = None
    lib.crc32c_build_table.argtypes = [_ADDRESS, _ADDRESS, _ADDRESS]
    lib.crc32c_slice8.restype = _UINT32
    lib.crc32c_slice8.argtypes = [_ADDRESS, _ADDRESS, _ADDRESS, _ADDRESS]
    lib.crc32c_slice16.restype = _UINT32
    lib.crc32c_slice16.argtypes = [_ADDRESS, _ADDRESS, _ADDRESS, _ADDRESS]
    lib.crc32c_combine.restype = _UINT32
    lib.crc32c_combine.argtypes = [_UINT32, _UINT32, _ADDRESS, _ADDRESS]
    return lib


lib = _load()


class _Tables:
    """Slicing tables for one polynomial, built by the Mojo kernel.

    The tables are owned here and are built once at import time so the folding
    loop never has to generate them.
    """

    def __init__(self, poly: int, n_slices: int):
        self.poly = poly
        self.n_slices = n_slices
        self.data = np.empty(n_slices * 256, dtype=np.uint32)
        addr = self.data.ctypes.data
        lib.crc32c_build_table(poly, addr, n_slices)
        self.ctypes = ctypes.c_uint8.from_buffer(self.data)
        self.addr = self.data.ctypes.data

    def fold(self, data, value: int, n: int) -> int:
        if self.n_slices == 8:
            return lib.crc32c_slice8(data, n, self.addr, value)
        return lib.crc32c_slice16(data, n, self.addr, value)


CRC32C_8 = _Tables(CRC32C_POLY, 8)
CRC32C_16 = _Tables(CRC32C_POLY, 16)
IEEE_8 = _Tables(IEEE_POLY, 8)
IEEE_16 = _Tables(IEEE_POLY, 16)


def _buffer(data) -> tuple:
    """Return (address, length) for any C-contiguous bytes-like object."""
    mv = memoryview(data)
    if not mv.contiguous:
        raise ValueError("input buffer must be C-contiguous")
    mv = mv.cast("B")
    if mv.nbytes == 0:
        return 0, 0
    # A memoryview has no usable address attribute, so wrap it once to reach
    # the buffer address. `from_buffer` keeps a reference to `mv`, so the
    # returned address stays valid for the call.
    return int(np.frombuffer(mv, dtype=np.uint8).ctypes.data), mv.nbytes


def crc32c(data, value: int = 0) -> int:
    """CRC-32C of `data`, continuing from `value`. Upstream's signature."""
    addr, n = _buffer(data)
    if n == 0:
        return value & 0xFFFFFFFF
    if n >= 1024:
        return CRC32C_16.fold(addr, value & 0xFFFFFFFF, n)
    return CRC32C_8.fold(addr, value & 0xFFFFFFFF, n)


def crc32c_width(data, value: int = 0, width: int = 16) -> int:
    """CRC-32C forced through the slice-by-`width` kernel.

    Both widths must agree with `crc32c`; this exists so the tests can check
    the two folding loops against each other and against upstream.
    """
    if width not in (8, 16):
        raise ValueError("width must be 8 or 16")
    addr, n = _buffer(data)
    if n == 0:
        return value & 0xFFFFFFFF
    tables = CRC32C_16 if width == 16 else CRC32C_8
    return tables.fold(addr, value & 0xFFFFFFFF, n)


def crc32(data, value: int = 0) -> int:
    """IEEE CRC-32 (zlib flavour), the other reflected CRC in the same family.

    Upstream `crc32c` exports a deprecated `crc32` that is an alias of
    `crc32c`; this one is the genuine IEEE polynomial, which is what a caller
    of this port usually wants alongside the Castagnoli one.
    """
    addr, n = _buffer(data)
    if n == 0:
        return value & 0xFFFFFFFF
    if n >= 1024:
        return IEEE_16.fold(addr, value & 0xFFFFFFFF, n)
    return IEEE_8.fold(addr, value & 0xFFFFFFFF, n)


def combine(crc1: int, crc2: int, len2: int, poly: int = CRC32C_POLY) -> int:
    """Merge the checksums of two consecutive buffers, zlib `crc32_combine`."""
    return lib.crc32c_combine(crc1, crc2, int(len2), int(poly))
