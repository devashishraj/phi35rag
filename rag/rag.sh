#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "# Running the rag.py"
if ! python rag.py; then
    log_error "rag.py failed to execute. Please check the script and inputs."
fi


echo "checking required directories exist"
if [[ ! -d ${OUTPUT_DIR}/output4/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output4/ does not exist."
fi

echo "moving output to next step"

#moving only relevant part to next stage
mv ragResponses.json ${OUTPUT_DIR}/output4/
