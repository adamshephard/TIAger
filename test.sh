#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=30g

echo "Building docker"
./build.sh

echo "Removing volume..."
docker volume rm tiager-ensemble
docker volume rm tiager-ensemble-tmp

echo "Creating volume..."
docker volume create tiager-ensemble
docker volume create tiager-ensemble-tmp
echo $SCRIPTPATH/testinput/
echo "Running algorithm..."

docker run --rm \
        --memory=$MEMORY \
        --memory-swap=$MEMORY \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        -v tiager-ensemble:/output/ \
        -v tiager-ensemble-tmp:/tempoutput/ \
        --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
        tiagerensemble

