#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t tiacentrealgorithmtilsmostafav2 "$SCRIPTPATH"
