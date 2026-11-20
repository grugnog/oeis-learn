"""Optional CPU observer entrypoint, explicitly mounted into a retained image."""
import sys
from pathlib import Path
import signal
import traceback

from run import load, write
from analysis import analyse, regressions, digest


def main():
    output = Path('/work/output')
    def interrupted(*_):
        raise KeyboardInterrupt()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        from oeis_learn.synthesis.client import SynthesisClient
        from oeis_learn.synthesis.admission import reference_identity
        from oeis_learn.experiments.source_snapshot import source_snapshot
        with SynthesisClient(digest('oeis-observer/v1'), cache_root=output / 'private/native') as client:
            write(output / 'runtime.json', dict(native=client.runtime_identity('bigint-body/v1'),
                reference=reference_identity(), source=digest(source_snapshot())))
            if sys.argv[1] == 'self-test':
                import unittest
                suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
                unit = unittest.TextTestRunner(verbosity=2).run(suite)
                if not unit.wasSuccessful():
                    raise ValueError('observer unit/fixture regressions failed')
                regressions(client, output)
            elif sys.argv[1] == 'audit':
                checks = regressions(client, output / 'private/self-test')
                write(output / 'regressions.json', checks)
                result = analyse(load(output / 'private/request.json', maximum=256 << 20), client, output)
                return 2 if any(r['status'] == 'INCOMPLETE' or r.get('baseline', {}).get('status') == 'MISMATCH' or any(
                    r.get(k, {}).get('status') in {'INCOMPLETE','EXECUTOR_DISAGREEMENT'}
                    for k in ('baseline','continuation')) for r in result['rows']) else 0
            else:
                raise ValueError('unknown CPU diagnostic operation')
        return 0
    except BaseException as exc:
        write(output / 'worker-failure.json', dict(error=type(exc).__name__, traceback=traceback.format_exc()))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
