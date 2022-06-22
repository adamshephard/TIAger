#!/usr/bin/env bash

./build.sh

docker save tiagerensemble | gzip -c > tiacentrealgorithmtils.tar.xz