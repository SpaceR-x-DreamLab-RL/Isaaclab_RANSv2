#!/bin/bash
if [[ -n "$DISPLAY" ]]; then
    DISPLAY_FLAG=(-e "DISPLAY=$DISPLAY")
else
    DISPLAY_FLAG=()
fi

CONTAINER_NAME="isaac-lab-ransv2"

docker exec \
    --interactive \
    --tty \
    "${DISPLAY_FLAG[@]}" \
    "${CONTAINER_NAME}" \
    bash

# docker exec --interactive --tty -e DISPLAY=$DISPLAY "${CONTAINER_NAME}" bash