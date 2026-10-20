"""Read-only current caller audit; historical dispositions are not regenerated."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check current source reachability')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from oeis_learn.synthesis.retirement import audit
    report = audit(ROOT / 'src/oeis_learn', project_root=ROOT)
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
