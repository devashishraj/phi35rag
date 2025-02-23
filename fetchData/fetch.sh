#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# checks if an environment variable is set and assigns a default value if it's not:
if [ -z "${Fetch_OUTPUT}" ]; then
    Fetch_OUTPUT="WikiRC.json"

echo "checking required directories exist"

# Ensure the next step directories exist
if [[ ! -d ${OUTPUT_DIR}/output1/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output1/ does not exist."
fi

if [[ ! -d  ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ ]]; then
    log_error "Directory ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ does not exist."
fi


# Run the data preparation script
echo "running getWikiChangesToJson.py script to fetch data"
if ! python getWikiChangesToJson.py; then
    log_error "getWikiChangesToJson.py failed to execute. Please check the script and inputs."
fi


mv $Fetch_OUTPUT ${INPUT_DIR}/input1/phi35ragRepo/queryGen/ 

#move output to sharedDir
echo "moving output to next step"

#moving only relevant part to next stage
mv ${INPUT_DIR}/input1/phi35ragRepo/queryGen ${OUTPUT_DIR}/output1/ 
mv ${INPUT_DIR}/input1/phi35ragRepo/dataEmbed ${OUTPUT_DIR}/output1/ 
mv ${INPUT_DIR}/input1/phi35ragRepo/rag ${OUTPUT_DIR}/output1/

# Final message
echo "Script completed successfully"