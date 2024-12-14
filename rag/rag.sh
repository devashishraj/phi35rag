# Run the data preparation script

echo "rag.py"
python rag.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output4 to store generated response"

#moving only relevant part to next stage
mv ${INPUT_DIR}}/input4/phi35ragRepo/rag/llmResponse.json ${OUTPUT_DIR}/ouput4/
