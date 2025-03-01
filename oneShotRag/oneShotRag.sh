#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "checking required directories exist"
if [[ ! -d ${OUTPUT_DIR}/output5/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output5/ does not exist."
fi


echo "# Running the oneShotRag.py"
if ! python3 oneShotRag.py; then
    log_error "oneShotRag failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"

#moving only relevant part to next stage
mv WikiRC_ESO.json ${INPUT_DIR}/input5/output4/factScore
mv ${INPUT_DIR}/input5/output4/factScore ${OUTPUT_DIR}/output5/