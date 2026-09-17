#!/usr/bin/env bash
podman build -t oeis .
podman save --format docker-archive oeis > oeis.tar
smolvm machine delete -f --name oeis
smolvm machine create --name oeis --smolfile Smolfile --volume "$(pwd)":/workspace
smolvm machine start --name oeis
