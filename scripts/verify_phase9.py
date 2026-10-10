"""Container phase-9 acceptance using actual executable gate replay."""

import argparse
from oeis_learn.evaluation.cli_acceptance import verify_discovery_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    return verify_discovery_cli(args.evidence)


if __name__ == "__main__":
    main()
