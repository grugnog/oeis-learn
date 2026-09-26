#!/usr/bin/env bash
set -e

cp /etc/nixos/packages/lab-gpu/lab-gpu* system/
cp /etc/nixos/pi/llama-swap.js system/
podman build --target smolvm -t oeis .
rm oeis.tar
podman save --format docker-archive -o oeis.tar oeis
smolvm machine delete -f --name oeis
smolvm machine create --name oeis --smolfile Smolfile --volume "$(pwd)":/workspace
smolvm machine start --name oeis
