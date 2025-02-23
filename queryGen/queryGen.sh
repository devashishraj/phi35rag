#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "checking required directories exist"
if [[ ! -d ${INPUT_DIR}/input2/output1/dataEmbed/ ]]; then
    log_error "Directory ${INPUT_DIR}/input2/output1/dataEmbed/ does not exist."
fi

if [[ ! -d ${INPUT_DIR}/input2/output1/rag/ ]]; then
    log_error "Directory ${INPUT_DIR}/input2/output1/rag/ does not exist."
fi

if [[ ! -d  ${OUTPUT_DIR}/output2/ ]]; then
    log_error "Directory  ${OUTPUT_DIR}/output2/ does not exist."
fi


echo "Running the query generation script"
if ! python queryGen.py; then
    log_error " queryGen.py failed to execute. Please check the script and inputs."
fi


echo "moving output to next step"

cp WikiRC_Q.json ${INPUT_DIR}/input2/output1/dataEmbed/ 
mv WikiRC_Q.json ${INPUT_DIR}/input2/output1/rag/

#moving only relevant part to next stage
mv ${INPUT_DIR}/input2/output1/dataEmbed ${OUTPUT_DIR}/output2/
mv ${INPUT_DIR}/input2/output1/rag ${OUTPUT_DIR}/output2/
