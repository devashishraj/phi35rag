#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}


echo "running getWikiChangesToJson.py script to fetch data"

# Run the data preparation script
echo "Running data_prep.py..."
if ! python getWikiChangesToJson.py; then
    log_error "getWikiChangesToJson.py failed to execute. Please check the script and inputs."
fi

# Ensure the target directories exist
if [[ ! -d ${OUTPUT_DIR}/output1/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output1/ does not exist."
fi

if [[ ! -d  ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ ]]; then
    log_error "Directory ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ does not exist."
fi


mv wikipedia_article_changes.json ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ || log_error "Failed to move wikipedia_article_changes.json"

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output1 to generate quereis for RAG "

#moving only relevant part to next stage
mv ${INPUT_DIR}/input1/phi35ragRepo/queryGen ${OUTPUT_DIR}/output1/ || log_error "Failed to move queryGen "
mv ${INPUT_DIR}/input1/phi35ragRepo/dataEmbed ${OUTPUT_DIR}/output1/ || log_error "Failed to move dataEmbed"
mv ${INPUT_DIR}/input1/phi35ragRepo/rag ${OUTPUT_DIR}/output1/ || log_error "Failed to move rag"

# Final message
echo "Script completed successfully"