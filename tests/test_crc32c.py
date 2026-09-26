"""Parity of the Mojo CRC kernels with the real `crc32c` C extension.

A CRC is integer arithmetic over bytes: there is no FMA and no rounding, so
every assertion here is exact equality. Anything looser would hide a wrong
table, a wrong shift or a dropped tail byte.
"""

import warnings

import numpy as np
import pytest

import mojo_crc32c as mcrc
from mojo_crc32c import _lib

crc32c = pytest.importorskip("crc32c")

# The check value every CRC-32C implementation agrees on.
CASTAGNOLI_CHECK = 0xE3069283
IEEE_CHECK = 0xCBF43926


def _buffers():
    rng = np.random.default_rng(20260926)
    yield b""
    yield b"a"
    yield b"123456789"
    yield b"\x00" * 64
    yield b"\xff" * 1024
    for n in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 255, 4096, 100_003):
        yield rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    yield (b"the quick brown fox " * 5000)[:65536]


BUFFERS = list(_buffers())


def test_check_value_matches_upstream():
    assert mcrc.crc32c(b"123456789") == CASTAGNOLI_CHECK
    assert crc32c.crc32c(b"123456789") == CASTAGNOLI_CHECK


@pytest.mark.parametrize("data", BUFFERS, ids=lambda b: f"len{len(b)}")
def test_matches_upstream_crc32c(data):
    assert mcrc.crc32c(data) == crc32c.crc32c(data)


@pytest.mark.parametrize("data", BUFFERS, ids=lambda b: f"len{len(b)}")
def test_slice_widths_agree_with_upstream(data):
    assert mcrc.crc32c_width(data, 0, 8) == crc32c.crc32c(data)
    assert mcrc.crc32c_width(data, 0, 16) == crc32c.crc32c(data)


@pytest.mark.parametrize("width", [8, 16])
def test_incremental_value_is_upstream_incremental(width):
    """`crc32c(tail, crc32c(head))` must equal the upstream single-shot value.

    This is the property a dropped complement or a wrong seed would break, and
    it exercises the `init` argument of the folding kernel.
    """
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, 9999, dtype=np.uint8).tobytes()
    for cut in (0, 1, 7, 8, 15, 16, 17, 4096, 5000, 9998, 9999):
        head, tail = data[:cut], data[cut:]
        seed = mcrc.crc32c_width(head, 0, width)
        assert mcrc.crc32c_width(tail, seed, width) == crc32c.crc32c(data)


def test_ieee_crc32_matches_zlib():
    import zlib

    for data in BUFFERS:
        assert mcrc.crc32(data) == zlib.crc32(data)


def test_ieeee_check_value():
    assert mcrc.crc32(b"123456789") == IEEE_CHECK


def test_crc32_incremental_matches_zlib():
    data = BUFFERS[-1]
    seed = mcrc.crc32(data[:1234])
    import zlib

    assert mcrc.crc32(data[1234:], seed) == zlib.crc32(data)


def test_table_matches_independent_construction():
    """The Mojo slicing tables must equal a straightforward Python build."""

    def build(poly, n_slices):
        t = [0] * 256
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ (poly if c & 1 else 0)
            t[i] = c
        for s in range(1, n_slices):
            prev = t[(s - 1) * 256:s * 256]
            t += [(c >> 8) ^ t[c & 0xFF] for c in prev]
        return np.array(t, dtype=np.uint32)

    np.testing.assert_array_equal(
        _lib.CRC32C_8.data, build(_lib.CRC32C_POLY, 8)
    )
    np.testing.assert_array_equal(
        _lib.CRC32C_16.data, build(_lib.CRC32C_POLY, 16)
    )
    np.testing.assert_array_equal(
        _lib.IEEE_16.data, build(_lib.IEEE_POLY, 16)
    )


def test_combine_matches_upstream_for_random_splits():
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(2, 40000))
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        cut = int(rng.integers(0, n + 1))
        head = mcrc.crc32c(data[:cut])
        tail = mcrc.crc32c(data[cut:])
        merged = mcrc.combine(head, tail, n - cut)
        assert merged == crc32c.crc32c(data)


def test_combine_with_empty_tail_is_identity():
    data = BUFFERS[5]
    head = mcrc.crc32c(data)
    assert mcrc.combine(head, 0, 0) == head
    assert mcrc.combine(head, mcrc.crc32c(b""), 0) == head


def test_combine_ieee_polynomial():
    import zlib

    data = BUFFERS[-1]
    cut = 4096
    merged = mcrc.combine(
        mcrc.crc32(data[:cut]), mcrc.crc32(data[cut:]), len(data) - cut, _lib.IEEE_POLY
    )
    assert merged == zlib.crc32(data)


@pytest.mark.parametrize("workers", [2, 3, 8, 16])
def test_threaded_matches_single_shot(workers):
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, 3_000_000, dtype=np.uint8).tobytes()
    got = mcrc.crc32c_threaded(data, 0, workers)
    assert got == crc32c.crc32c(data)


def test_threaded_honours_seed():
    data = BUFFERS[-1] * 20
    seed = mcrc.crc32c(b"prefix")
    assert mcrc.crc32c_threaded(data, seed, 8) == crc32c.crc32c(data, seed)


def test_threaded_falls_back_on_small_input():
    data = b"small buffer"
    assert mcrc.crc32c_threaded(data, 0, 8) == crc32c.crc32c(data)


def test_numpy_and_memoryview_inputs():
    arr = np.frombuffer(BUFFERS[-1], dtype=np.uint8)
    assert mcrc.crc32c(arr) == crc32c.crc32c(arr.tobytes())
    assert mcrc.crc32c(memoryview(arr)) == crc32c.crc32c(arr.tobytes())
    floats = np.arange(1000, dtype=np.float64)
    assert mcrc.crc32c(floats) == crc32c.crc32c(floats.tobytes())


def test_hash_wrapper_matches_upstream():
    data = BUFFERS[-1]
    mine = mcrc.CRC32CHash()
    mine.update(data[:1000])
    mine.update(data[1000:])
    theirs = crc32c.CRC32CHash()
    theirs.update(data[:1000])
    theirs.update(data[1000:])
    assert mine.hexdigest() == theirs.hexdigest()
    assert mine.digest() == theirs.digest()
    assert mine.checksum == theirs.checksum
    assert mine.digest_size == theirs.digest_size
    assert mine.block_size == theirs.block_size
    assert mine.name == theirs.name
    assert mcrc.CRC32CHash(data).hexdigest() == crc32c.CRC32CHash(data).hexdigest()


def test_hash_copy_keeps_state():
    h = mcrc.CRC32CHash(b"abc")
    c = h.copy()
    h.update(b"def")
    assert c.hexdigest() == mcrc.crc32c(b"abc").to_bytes(4, "big").hex()
    assert h.hexdigest() == mcrc.crc32c(b"abcdef").to_bytes(4, "big").hex()


def test_deprecated_crc32_alias_is_the_same_polynomial():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert crc32c.crc32(b"123456789") == CASTAGNOLI_CHECK
    assert mcrc.crc32c(b"123456789") == CASTAGNOLI_CHECK
