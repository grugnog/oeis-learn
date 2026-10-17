"""Plan and load shared CI images without changing the public container wrapper."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

TARGETS = ("qualification", "gpu")
BUILD_INPUTS = (
    "Dockerfile",
    ".dockerignore",
    "docker/dev",
    "pyproject.toml",
    "uv.lock",
    "setup.py",
    "MANIFEST.in",
    "crates/oeis_wasm_evaluator",
)


def plan(env, event):
    target = env["CI_IMAGE_TARGET"]
    if target not in TARGETS:
        raise ValueError("unknown image target")
    repository = env["GITHUB_REPOSITORY"].lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*", repository):
        raise ValueError("invalid repository name")
    run = env["GITHUB_RUN_ID"] + "-" + env["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"\d+-\d+", run):
        raise ValueError("invalid run identity")
    event_name = env["GITHUB_EVENT_NAME"]
    # Fork and Dependabot PR tokens cannot publish packages. Never use a
    # pull_request_target/workflow_run handoff to grant untrusted code writes.
    if event_name not in {"pull_request", "push", "workflow_dispatch"}:
        raise ValueError("unsupported image build event")
    publish = env["GITHUB_ACTOR"] != "dependabot[bot]" and (
        event_name != "pull_request"
        or event["pull_request"]["head"]["repo"]["full_name"].lower() == repository
    )
    image = "ghcr.io/" + repository
    cache = image + ":buildcache-" + target + "-main"
    cache_to = "type=gha,scope=oeis-" + target + ",mode=max,ignore-error=true"
    default_ref = "refs/heads/" + event["repository"]["default_branch"]
    # PRs can publish only their uniquely named images and PR-scoped GHA cache;
    # only a default-branch push/manual run refreshes the shared registry cache.
    is_default = publish and event_name != "pull_request" and env["GITHUB_REF"] == default_ref
    tag = image + ":ci-" + target + "-" + run
    if is_default:
        cache_to += "\ntype=registry,ref=" + cache + ",mode=max"
    return dict(
        publish=str(publish).lower(),
        transport="registry" if publish else "archive",
        image=image,
        tag=tag,
        tags=tag + ("\n" + image + ":" + target + "-main" if is_default else ""),
        artifact="ci-image-" + target,
        cache_from="type=registry,ref=" + cache + "\ntype=gha,scope=oeis-" + target,
        cache_to=cache_to,
    )


def recipe_changed(base, revision):
    result = subprocess.run(["git", "diff", "--quiet", base, revision, "--", *BUILD_INPUTS])
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot determine container build changes")
    return result.returncode == 1


def load_image(engine, transport, reference, alias, revision, archive=None, token=None):
    if engine not in ("docker", "podman") or transport not in ("registry", "archive"):
        raise ValueError("invalid engine or image transport")
    suffix = r"@sha256:[0-9a-f]{64}" if transport == "registry" else r":[a-z0-9][a-z0-9_.-]*"
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9_./-]*" + suffix, reference):
        raise ValueError("registry images require an immutable digest; archives require a tag")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", alias):
        raise ValueError("invalid local image alias")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("invalid source revision")
    if transport == "registry":
        logged_in = False
        try:
            if token:
                subprocess.run(
                    [
                        engine,
                        "login",
                        "ghcr.io",
                        "--username",
                        os.environ["GITHUB_ACTOR"],
                        "--password-stdin",
                    ],
                    input=token + "\n",
                    text=True,
                    check=True,
                )
                logged_in = True
            subprocess.run([engine, "pull", reference], check=True)
        finally:
            if logged_in:
                subprocess.run([engine, "logout", "ghcr.io"], check=True)
    else:
        if archive is None or not Path(archive).is_file():
            raise ValueError("image archive is missing")
        subprocess.run([engine, "load", "--input", str(archive)], check=True)
    result = subprocess.run(
        [
            engine,
            "image",
            "inspect",
            reference,
            "--format",
            '{{ index .Config.Labels "org.opencontainers.image.revision" }}',
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip() != revision:
        raise ValueError(
            f"image source revision does not match the tested checkout: "
            f"expected {revision}, got {result.stdout.strip()!r}"
        )
    subprocess.run([engine, "tag", reference, alias], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    load = commands.add_parser("load")
    load.add_argument("--engine", choices=("docker", "podman"), required=True)
    load.add_argument("--transport", choices=("registry", "archive"), required=True)
    load.add_argument("--reference", required=True)
    load.add_argument("--alias", required=True)
    load.add_argument("--revision", required=True)
    load.add_argument("--archive")
    args = parser.parse_args()
    if args.command == "load":
        load_image(
            args.engine,
            args.transport,
            args.reference,
            args.alias,
            args.revision,
            args.archive,
            os.environ.get("GHCR_TOKEN"),
        )
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    value = plan(os.environ, event)
    changed = True
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
        changed = recipe_changed(event["pull_request"]["base"]["sha"], os.environ["GITHUB_SHA"])
    value["podman_build"] = str(changed).lower()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, item in value.items():
            stream.write(f"{key}<<OEIS_OUTPUT\n{item}\nOEIS_OUTPUT\n")


if __name__ == "__main__":
    main()
