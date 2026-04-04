"""Ensure the QFun repo root is on sys.path.

Run from any notebook in this directory:
    %run _path_setup
"""

import sys
from pathlib import Path

for _p in (Path.cwd(), Path.cwd().parent):
    if (_p / "qfun").is_dir():
        _root = str(_p.resolve())
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break
