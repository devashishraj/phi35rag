# Run the data embedding script to create vectorStore

echo "dataEmbed.py"
python dataEmbed.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output3/ to perform retrieval-augmented generation"

# moving files to next stage folder
mv vectorstore_index.faiss ${INPUT_DIR}/input3/output2/rag/

#moving only next statge folder(s) to output
mv ${INPUT_DIR}/input3/output2/rag ${OUTPUT_DIR}/output3/
