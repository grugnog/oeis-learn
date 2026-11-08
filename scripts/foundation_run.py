#!/usr/bin/env python3
"""Portable Docker/Podman foundation and discovery commands; stdlib-only host."""

from foundation_local import entrypoint

if __name__ == "__main__":
    raise SystemExit(entrypoint())
