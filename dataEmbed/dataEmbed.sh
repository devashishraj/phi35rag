#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "# Running the dataEmbed.py"
if ! python dataEmbed.py; then
    log_error " dataEmbed.py failed to execute. Please check the script and inputs."
fi

echo "checking required directories exist"
if [[ ! -d ${INPUT_DIR}/input3/output2/rag/ ]]; then
    log_error "Directory ${INPUT_DIR}/input3/output2/rag/ does not exist."
fi

if [[ ! -d ${OUTPUT_DIR}/output3/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output3/ does not exist."
fi

echo "moving output to next step"

# moving files to next stage folder
mv vectorstore_index.faiss ${INPUT_DIR}/input3/output2/rag/

#moving only next statge folder(s) to output
mv ${INPUT_DIR}/input3/output2/rag ${OUTPUT_DIR}/output3/
