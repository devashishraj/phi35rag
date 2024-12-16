# Run the query generation script

echo "queryGen.py"
python queryGen.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output2 to create vectorDB of data"

mv wikipedia_article_changes.json ${INPUT_DIR}/input2/output1/dataEmbed/
mv ragQueries.json ${INPUT_DIR}/input2/output1/rag/

#moving only relevant part to next stage
mv ${INPUT_DIR}/input2/output1/dataEmbed ${OUTPUT_DIR}/output2/
mv ${INPUT_DIR}/input2/output1/rag ${OUTPUT_DIR}/output2/
