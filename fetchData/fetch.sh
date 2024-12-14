# Run the data preparation script

echo "getWikiChangesToJson.py"
python getWikiChangesToJson.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/step to generate quereis for RAG "

mv wikipedia_article_changes.json ${INPUT_DIR}/input1/phi35ragRepo/queryGen/

#moving only relevant part to next stage
mv ${INPUT_DIR}}/input1/phi35ragRepo/queryGen/ ${OUTPUT_DIR}/ouput1/
mv ${INPUT_DIR}}/input1/phi35ragRepo/dataEmbed/ ${OUTPUT_DIR}/output1/
mv ${INPUT_DIR}}/input1/phi35ragRepo/rag/ ${OUTPUT_DIR}/ouput1/
