"""CRC-32C (Castagnoli) with slice-by-8 and slice-by-16 table folding.

The upstream `crc32c` distribution is a C extension: `crc32c.crc32c(data,
value)` is `crc ^= 0xffffffff; result = crc_fn(...); result ^= 0xffffffff` with
a reflected table CRC-32C (polynomial 0x82f63b78) and an SSE4.2 / ARMv8-CRC
hardware path when the host supports it. The whole compute core of the package
is that one reflected table-driven loop, so that is what is ported here.

Exported:

* `crc32c_build_table` builds the slicing tables for an arbitrary reflected
  polynomial, so the same folding kernels also serve IEEE CRC-32 (zlib) and any
  other reflected CRC. Table construction is part of the ported surface rather
  than a pasted-in constant array, and the tests check it against an
  independent construction.
* `crc32c_slice8` / `crc32c_slice16` fold `n` bytes through those tables.
* `crc32c_combine` is zlib's `crc32_combine`: the GF(2)[x]/p operator that
  merges the register state of two consecutive chunks, which is what lets the
  Python shim split one buffer across a thread pool and still produce the
  single-shot checksum.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.utils import StaticTuple

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def up(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


@export("crc32c_build_table")
def crc32c_build_table(poly: Int, table_addr: Int, n_slices: Int) abi("C"):
    """Fill `n_slices` * 256 reflected slicing tables at `table_addr`.

    `t[0]` is the plain byte table for the reflected `poly`; `t[k]` is the
    table that folds the byte shifted left by `k` bytes, built by the standard
    recurrence `t[k][i] = (t[k-1][i] >> 8) ^ t[0][t[k-1][i] & 0xff]`.
    """
    var t = up(table_addr)
    for i in range(256):
        var c = i
        for _ in range(8):
            if c & 1:
                c = (c >> 1) ^ poly
            else:
                c = c >> 1
        t[unsafe_offset=i] = UInt32(c)
    for s in range(1, n_slices):
        var base = s * 256
        var prev = base - 256
        for i in range(256):
            var c = Int(t[unsafe_offset=prev + i])
            t[unsafe_offset=base + i] = UInt32((c >> 8) ^ Int(t[unsafe_offset=c & 0xFF]))


@export("crc32c_slice8")
def crc32c_slice8(
    data_addr: Int, n: Int, table_addr: Int, init: Int
) abi("C") -> UInt32:
    """Slice-by-8 CRC over `n` bytes, continuing from the upstream `init` value.

    The `crc ^= 0xffffffff` bracketing is upstream's convention and is what
    makes `crc32c(data, crc32c(head)) == crc32c(head + tail)` hold.
    """
    var p = bp(data_addr)
    var t = up(table_addr)
    var t1 = t.unsafe_offset(256)
    var t2 = t.unsafe_offset(512)
    var t3 = t.unsafe_offset(768)
    var t4 = t.unsafe_offset(1024)
    var t5 = t.unsafe_offset(1280)
    var t6 = t.unsafe_offset(1536)
    var t7 = t.unsafe_offset(1792)
    var crc = UInt32(init) ^ UInt32(0xFFFFFFFF)
    var i = 0
    while i + 8 <= n:
        var w = UInt32(p[unsafe_offset=i]) | (
            UInt32(p[unsafe_offset=i + 1]) << 8
        ) | (UInt32(p[unsafe_offset=i + 2]) << 16) | (UInt32(p[unsafe_offset=i + 3]) << 24)
        crc ^= w
        crc = (
            t7[unsafe_offset=Int(crc & 0xFF)]
            ^ t6[unsafe_offset=Int((crc >> 8) & 0xFF)]
            ^ t5[unsafe_offset=Int((crc >> 16) & 0xFF)]
            ^ t4[unsafe_offset=Int(crc >> 24)]
            ^ t3[unsafe_offset=Int(p[unsafe_offset=i + 4])]
            ^ t2[unsafe_offset=Int(p[unsafe_offset=i + 5])]
            ^ t1[unsafe_offset=Int(p[unsafe_offset=i + 6])]
            ^ t[unsafe_offset=Int(p[unsafe_offset=i + 7])]
        )
        i += 8
    while i < n:
        crc = (
            t[unsafe_offset=Int((crc ^ UInt32(p[unsafe_offset=i])) & 0xFF)]
            ^ (crc >> 8)
        )
        i += 1
    return crc ^ UInt32(0xFFFFFFFF)


@export("crc32c_slice16")
def crc32c_slice16(
    data_addr: Int, n: Int, table_addr: Int, init: Int
) abi("C") -> UInt32:
    """Slice-by-16 CRC over `n` bytes; `table_addr` holds 16 * 256 entries."""
    var p = bp(data_addr)
    var t = up(table_addr)
    var t1 = t.unsafe_offset(256)
    var t2 = t.unsafe_offset(512)
    var t3 = t.unsafe_offset(768)
    var t4 = t.unsafe_offset(1024)
    var t5 = t.unsafe_offset(1280)
    var t6 = t.unsafe_offset(1536)
    var t7 = t.unsafe_offset(1792)
    var t8 = t.unsafe_offset(2048)
    var t9 = t.unsafe_offset(2304)
    var t10 = t.unsafe_offset(2560)
    var t11 = t.unsafe_offset(2816)
    var t12 = t.unsafe_offset(3072)
    var t13 = t.unsafe_offset(3328)
    var t14 = t.unsafe_offset(3584)
    var t15 = t.unsafe_offset(3840)
    var crc = UInt32(init) ^ UInt32(0xFFFFFFFF)
    var i = 0
    while i + 16 <= n:
        # The 16 bytes fold as one XOR of 16 table terms: the incoming
        # register occupies table positions 15..12, so its four bytes are
        # XORed with the first four data bytes, while positions 11..0 read
        # the data bytes directly.
        crc = (
            t15[unsafe_offset=Int((crc ^ UInt32(p[unsafe_offset=i])) & 0xFF)]
            ^ t14[unsafe_offset=Int(((crc >> 8) ^ UInt32(p[unsafe_offset=i + 1])) & 0xFF)]
            ^ t13[unsafe_offset=Int(((crc >> 16) ^ UInt32(p[unsafe_offset=i + 2])) & 0xFF)]
            ^ t12[unsafe_offset=Int(((crc >> 24) ^ UInt32(p[unsafe_offset=i + 3])) & 0xFF)]
            ^ t11[unsafe_offset=Int(p[unsafe_offset=i + 4])]
            ^ t10[unsafe_offset=Int(p[unsafe_offset=i + 5])]
            ^ t9[unsafe_offset=Int(p[unsafe_offset=i + 6])]
            ^ t8[unsafe_offset=Int(p[unsafe_offset=i + 7])]
            ^ t7[unsafe_offset=Int(p[unsafe_offset=i + 8])]
            ^ t6[unsafe_offset=Int(p[unsafe_offset=i + 9])]
            ^ t5[unsafe_offset=Int(p[unsafe_offset=i + 10])]
            ^ t4[unsafe_offset=Int(p[unsafe_offset=i + 11])]
            ^ t3[unsafe_offset=Int(p[unsafe_offset=i + 12])]
            ^ t2[unsafe_offset=Int(p[unsafe_offset=i + 13])]
            ^ t1[unsafe_offset=Int(p[unsafe_offset=i + 14])]
            ^ t[unsafe_offset=Int(p[unsafe_offset=i + 15])]
        )
        i += 16
    while i < n:
        crc = (
            t[unsafe_offset=Int((crc ^ UInt32(p[unsafe_offset=i])) & 0xFF)]
            ^ (crc >> 8)
        )
        i += 1
    return crc ^ UInt32(0xFFFFFFFF)


@export("crc32c_combine")
def crc32c_combine(
    crc1: UInt32, crc2: UInt32, len2: Int, poly: Int
) abi("C") -> UInt32:
    """Merge two consecutive CRCs over GF(2)[x]/poly.

    `crc1` is the checksum of everything before the split, `crc2` the checksum
    of the trailing `len2` bytes, both in upstream's output convention.

    Folding the tail from the head's register state differs from folding it
    from a fresh one by exactly the operator for `len2` zero bytes applied to
    `crc1` (the register states differ by `crc1` itself, since both are
    complemented), so the concatenation's checksum is `crc2 ^ R(crc1)`.
    """
    if len2 <= 0:
        return crc1
    # base starts as the operator for one zero bit: base[0] is x mod p, and
    # base[n] shifts a power of x down one place, which is what the reflected
    # register layout needs. Three squarings make it the one-zero-byte
    # operator, and each further squaring doubles the number of bytes.
    var base = StaticTuple[UInt32, 32]()
    base[0] = UInt32(poly)
    var row = UInt32(1)
    for n in range(1, 32):
        base[n] = row
        row = row << 1
    for _ in range(3):
        base = _square(base)
    # acc is the identity until a bit of len2 picks up a doubling.
    var acc = StaticTuple[UInt32, 32]()
    for n in range(32):
        acc[n] = UInt32(1) << UInt32(n)
    var rest = len2
    while rest != 0:
        if rest & 1:
            acc = _compose(base, acc)
        rest = rest >> 1
        base = _square(base)
    return crc2 ^ _times(acc, crc1)


def _times(mat: StaticTuple[UInt32, 32], vec: UInt32) -> UInt32:
    """Multiply the polynomial `vec` by the operator `mat`."""
    var sum = UInt32(0)
    var v = vec
    var idx = 0
    while v != 0:
        if v & 1:
            sum = sum ^ mat[idx]
        v = v >> 1
        idx += 1
    return sum


def _square(mat: StaticTuple[UInt32, 32]) -> StaticTuple[UInt32, 32]:
    """The square of every operator in `mat`."""
    var square = StaticTuple[UInt32, 32]()
    for n in range(32):
        square[n] = _times(mat, mat[n])
    return square


def _compose(outer: StaticTuple[UInt32, 32], inner: StaticTuple[UInt32, 32]) -> StaticTuple[UInt32, 32]:
    """The operator that applies `inner` and then `outer`."""
    var composed = StaticTuple[UInt32, 32]()
    for n in range(32):
        composed[n] = _times(outer, inner[n])
    return composed
