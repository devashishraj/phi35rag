# Run the query generation script

echo "getWikiChangesToJson.py"
python queryGen.sh

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output2 to create vectorDB of data"

mv wikipedia_article_changes.json ${INPUT_DIR}/input2/dataEmbed/
mv ragQueries.json ${INPUT_DIR}/input2/rag/

#moving only relevant part to next stage
mv ${INPUT_DIR}}/input2/phi35ragRepo/dataEmbed ${OUTPUT_DIR}/output2/
mv ${INPUT_DIR}}/input2/phi35ragRepo/rag ${OUTPUT_DIR}/output2/
