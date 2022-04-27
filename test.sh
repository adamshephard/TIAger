#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=16g

echo "Building docker"
./build.sh

echo "Removing volume..."
docker volume rm tiger-tils-output-mostafa-v2
docker volume rm tiger-tils-output-mostafa-v2-tmp

echo "Creating volume..."
docker volume create tiger-tils-output-mostafa-v2
docker volume create tiger-tils-output-mostafa-v2-tmp
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
        -v tiger-tils-output-mostafa-v2:/output/ \
        --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
        tiacentrealgorithmtilsmostafav2
#         -v tiger-tils-output-mostafa-v2-tmp:/tempoutput/ \

# echo "Checking output files..."
# docker run --rm \
#         -v tiger-tils-output-ruoyu:/output/ \
#         python:3.6-slim \
#         python -m json.tool $DETECTION_FILE; \
#         [[ -f $SEGMENTATION_FILE ]] || printf 'Expected file %s does not exist!\n' "$SEGMENTATION_FILE"; \
#         [[ -f $TILS_SCORE_FILE ]] || printf 'Expected file %s does not exist!\n' "$TILS_SCORE_FILE"; \

# echo "Removing volume..."
# docker volume rm tiger-tils-output-ruoyu
