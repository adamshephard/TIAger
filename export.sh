#!/usr/bin/env bash

./build.sh

docker save tiacentrealgorithmtilsmostafav2 | gzip -c > tiacentrealgorithmtils.tar.xz