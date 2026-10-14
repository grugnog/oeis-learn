#!/usr/bin/env python3
"""Portable Docker/Podman foundation commands. Host Python uses the standard library."""

from foundation_local import entrypoint

if __name__ == "__main__":
    raise SystemExit(entrypoint())
