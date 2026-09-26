import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-crc32c.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-crc32c.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )
