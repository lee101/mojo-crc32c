# mojo-crc32c

`mojo-crc32c` is the compute core of
[crc32c](https://pypi.org/project/crc32c/) with the CRC loop written in Mojo:
a reflected table CRC-32C (Castagnoli, polynomial `0x82f63b78`) folded
slice-by-8 or slice-by-16 at a time, plus the GF(2) combine operator that
merges the checksums of two consecutive buffers.

The Python package is `mojo_crc32c`, so it installs alongside the real `crc32c`
and the tests compare the two directly.

```python
import mojo_crc32c as mcrc

mcrc.crc32c(b"123456789")            # 0xe3069283
mcrc.crc32(b"123456789")             # 0xcbf43926, the IEEE CRC-32
h = mcrc.CRC32CHash()
h.update(b"1234"); h.update(b"56789")
h.hexdigest()                        # 'e3069283'
```

## What the real package is

`crc32c` is a small C extension (`ext/_crc32c.c`) whose entire public surface
is:

| upstream | what it does |
| --- | --- |
| `crc32c.crc32c(data, value=0)` | CRC-32C, `crc ^= 0xffffffff` in and out, incremental through `value` |
| `crc32c.crc32(...)` | deprecated alias of `crc32c` |
| `crc32c.CRC32CHash` | `hashlib`-shaped wrapper with `update`/`digest`/`hexdigest`/`copy` |
| `crc32c.hardware_based` | whether an SSE4.2 / ARMv8 CRC instruction was found |
| `crc32c.big_endian` | build-time endianness of the extension |
| `python -m crc32c` | a file checksumming CLI |

`crc32c_inline` is `crc ^= 0xffffffff; result = crc_fn(crc, buf, len);
result ^= 0xffffffff`, where `crc_fn` is either the SSE4.2 `crc32` instruction,
the ARMv8 `crc32cx` instruction, or the table loop in `ext/crc32c_sw.c`. That
loop is the only arithmetic in the package, and it is what is ported here.

## Covered subset

| area | implemented API |
| --- | --- |
| CRC-32C | `crc32c(data, value=0)`, incremental through `value` |
| CRC-32C, chosen kernel | `crc32c_width(data, value=0, width=8 or 16)` |
| IEEE CRC-32 | `crc32(data, value=0)` |
| hashlib wrapper | `CRC32CHash` with `digest_size`, `block_size`, `name`, `checksum`, `update`, `digest`, `hexdigest`, `copy` |
| chunk merging | `combine(crc1, crc2, len2)` and the threaded `crc32c_threaded(data, value=0, workers=8)` |
| kernels | `crc32c_build_table`, `crc32c_slice8`, `crc32c_slice16`, `crc32c_combine` |

Not implemented, and why:

* **`hardware_based` and `big_endian`.** These are build-configuration
  constants of the C extension. The Mojo kernels are always a table fold and
  always read bytes little-endian by explicit byte loads, so there is no
  honest value to report; the attributes are deliberately absent rather than
  faked.
* **The `-M/-b/-s/-N` CLI** (`_cli.py`). Argument parsing and file iteration,
  no arithmetic; use the real `python -m crc32c`.
* **The SSE4.2 / ARMv8 CRC instruction path.** The tables are the portable
  implementation; issuing the instruction would need inline assembly, which
  `mojo build --emit shared-lib` has no story for here. That is exactly why the
  benchmark below is a loss against upstream.
* **`crc32c.crc32` as an alias of `crc32c`.** This port's `crc32` is the
  genuine IEEE CRC-32 instead, since a second name for the same Castagnoli sum
  would be dead weight. The parity test asserts the upstream alias really is
  the Castagnoli value.

The numerical contract is exact: a CRC is integer arithmetic over bytes, so
every test asserts equality, never a tolerance.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-crc32c.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. The parity tests additionally need
the real `crc32c` package installed, and compare against it byte for byte.

## Performance

Best-of-five wall clock on a 64 MiB random buffer, same process. Every case
verifies the checksum against the real extension before timing. The host is a
shared 72-core box with other builds running, so the absolute numbers move by
2-3x between runs; both columns move together.

| case | reference | GB/s | mojo-crc32c | GB/s | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| crc32c slice-by-8 | 10.5 ms | 6.42 | 57.2 ms | 1.17 | 0.18x |
| crc32c slice-by-16 | 12.2 ms | 5.52 | 85.7 ms | 0.78 | 0.14x |
| crc32c threaded x2 | 9.1 ms | 7.39 | 45.5 ms | 1.47 | 0.20x |
| crc32c threaded x4 | 23.7 ms | 2.83 | 58.5 ms | 1.15 | 0.41x |
| crc32c threaded x8 | 12.8 ms | 5.23 | 47.2 ms | 1.42 | 0.27x |
| crc32c threaded x16 | 17.3 ms | 3.88 | 52.2 ms | 1.28 | 0.33x |
| crc32 (IEEE) vs `zlib.crc32` | 59.3 ms | 1.13 | 64.2 ms | 1.05 | 0.92x |

The reference column for CRC-32C is the upstream C extension, which on this
host dispatches to the SSE4.2 `crc32` instruction. That instruction retires
three bytes per cycle, which no amount of table folding matches; the honest
result is that this port is 5-7x slower than upstream at the one job upstream
was built for, and the README says so rather than the other way round.

The IEEE row is the interesting one: `zlib.crc32` in this build is a table
loop too, and there the port is at parity (0.92x, inside the run-to-run noise),
which is what a bandwidth- and L1-bound fold should give.

Threading does not pay here, and `crc32c_threaded` is kept only because the
combine operator is a genuine ported primitive with its own parity tests. The
fold is L1-bound, the workers contend on a loaded host, and the hand-off costs
more than the fold: 1 MiB threaded x8 measured 0.9 ms against 0.9 ms serial, and
64 MiB x8 was slower than serial. The default `crc32c` therefore stays serial.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-crc32c.so`.

`crc32c_build_table` fills `n * 256` slicing tables for a reflected
polynomial, so the same folding kernels serve CRC-32C and IEEE CRC-32:

```
t[0][i] = i folded through the polynomial eight times
t[k][i] = (t[k-1][i] >> 8) ^ t[0][t[k-1][i] & 0xff]
```

`crc32c_slice8` folds eight bytes per step and `crc32c_slice16` sixteen. The
16-byte form is one XOR of sixteen table terms: the incoming register occupies
table positions 15..12, so its four bytes are XORed with the first four data
bytes, and positions 11..0 read the data bytes directly. Chaining the two
halves through the register instead (fold eight, then fold the next eight with
a second `crc ^= word`) is the form that circulates in a lot of code, and it
is wrong; the tests pin the correct one against the real extension at every
length from 0 to 100003.

`crc32c_combine` merges two consecutive checksums. Folding a tail from the
head's register state instead of a fresh one shifts the result by the operator
for `len2` zero bytes applied to `crc1` (the two register states differ by
`crc1` itself, because both returned values are complemented), so the answer is
`crc2 ^ R(crc1)`. `R` is built by squaring the one-zero-bit operator: three
squarings give one zero byte, and the set bits of `len2` select which doublings
to compose. That is GF(2)[x]/p arithmetic, 32x32 bit matrices, and it is what
makes the threaded path bit-identical to the serial one.

Buffers cross the C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[UInt8, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. The slicing tables are built once at import time into NumPy
arrays owned by `python/mojo_crc32c/_lib.py`.

## License

MIT
