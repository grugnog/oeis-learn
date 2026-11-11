"""Installed CPU corpus worker; invoked by the canonical host data command."""
from pathlib import Path
import signal
import sys
import traceback

from .common import read, write
from .corpus import prepare


def main():
    config = read(sys.argv[1])
    def interrupted(*_):
        raise KeyboardInterrupt()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        result = prepare(Path.cwd(), config)
        return 0 if result['status'] == 'COMPLETE' else 2
    except BaseException as exc:
        write(Path(config['cache']) / 'diagnostics/worker-failure.json',
              dict(error=type(exc).__name__, traceback=traceback.format_exc()))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
