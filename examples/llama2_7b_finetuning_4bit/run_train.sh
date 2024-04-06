#!/usr/bin/env bash

# Fail fast if an error occurs
set -e
export HUGGING_FACE_HUB_TOKEN="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"

# Get the directory of this script, which contains the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Train
ludwig train --config ${SCRIPT_DIR}/llama2_7b_4bit.yaml --dataset ludwig://alpaca
