#!/usr/bin/env bash
podman build -t oeis .
podman save --format docker-archive oeis > oeis.tar
smolvm machine delete -f --name oeis
smolvm machine create --name oeis --smolfile Smolfile --volume "$(pwd)":/workspace
smolvm machine start --name oeis
smolvm machine exec --name oeis -- sh -c 'mkdir -p /storage/docker /var/lib/docker /storage/containerd /var/lib/containerd
mount --bind /storage/docker /var/lib/docker
mount --bind /storage/containerd /var/lib/containerd
rm -f /var/run/docker.pid
dockerd --storage-driver=overlay2 >/tmp/dockerd.log 2>&1 &
for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info | grep "Storage Driver"'

