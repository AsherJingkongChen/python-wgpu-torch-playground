#! /usr/bin/env python3

"""
Build the project (Py)

## Usage

1.
```shell
./scripts/x_build_py.py
```

2.
```shell
python3 scripts/x_build_py.py
```
"""

from os import PathLike
from pathlib import Path


def build(env_dir: PathLike[str] | str | None = None) -> None:
    from subprocess import check_call
    from u_env import Env
    from x_clear import remove_globs

    env = Env(env_dir)
    python = env.data.executable

    remove_globs(*get_build_paths())
    check_call(
        f"{python} -m build --no-isolation --outdir=dist --wheel".split()
    )
    remove_globs("build/")
    check_call(f"{python} -m twine check --strict".split() + get_build_paths())
    check_call(
        f"{python} -m pip install --force-reinstall --no-deps".split()
        + get_build_paths()
    )


def get_build_paths() -> list[Path]:
    return list(Path("dist").glob("*"))


if __name__ == "__main__":
    from sys import argv

    env_dir = argv[1] if len(argv) > 1 else None
    build(env_dir)
