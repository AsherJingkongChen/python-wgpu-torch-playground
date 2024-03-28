#! /usr/bin/env python3

"""
Clear artifacts

## Usage

1.
```shell
./scripts/x_clear.py
```

2.
```shell
python3 scripts/x_clear.py
```
"""


from os import PathLike


def clear(env_dir: PathLike[str] | str | None = None) -> None:
    from u_env import Env

    remove_globs(
        Env.resolve_dir(env_dir),
        "dist/",
        "python/**/*.so",
        "target/",
        ".pytest_cache/",
        "*.egg-info",
        "**/__pycache__",
    )


def remove_globs(*patterns: PathLike[str] | str) -> None:
    """
    Remove all directories or files that match the glob patterns

    ## Note
    - It uses `pathlib.Path.glob` to expand the glob patterns
    """
    from glob import iglob
    from os import unlink
    from os.path import exists, isdir
    from shutil import rmtree

    def expand(pattern: PathLike[str] | str):
        return iglob(str(pattern), recursive=True)

    for glob in map(expand, patterns):
        for path in glob:
            if isdir(path):
                rmtree(path)
            elif not exists(path):
                continue
            else:
                unlink(path)


if __name__ == "__main__":
    from sys import argv

    clear(argv[1] if len(argv) > 1 else None)
