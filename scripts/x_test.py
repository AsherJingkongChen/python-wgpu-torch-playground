#! /usr/bin/env python3

"""
Run tests

## Usage

1.
```shell
./scripts/x_test.py
```

2.
```shell
python3 scripts/x_test.py
```
"""

from os import PathLike
from pathlib import Path


def test(env_dir: PathLike[str] | str | None = None) -> None:
    from subprocess import check_call
    from u_env import Env

    env = Env(env_dir)
    python = env.data.executable

    test_paths = list(Path().glob("**/__tests__/**/*.py"))
    check_call(
        f"{python} -m pytest --capture=no --exitfirst "
        "--ignore=*-packages --import-mode=append ".split() + test_paths
    )


if __name__ == "__main__":
    from sys import argv

    env_dir = argv[1] if len(argv) > 1 else None
    test(env_dir)
