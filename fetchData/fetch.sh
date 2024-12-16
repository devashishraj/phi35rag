# Run the data fetch script

echo "running getWikiChangesToJson.py script to fetch data"
python getWikiChangesToJson.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/output1 to generate quereis for RAG "

mv wikipedia_article_changes.json ${INPUT_DIR}/input1/phi35ragRepo/queryGen/

#moving only relevant part to next stage
mv ${INPUT_DIR}/input1/phi35ragRepo/queryGen ${OUTPUT_DIR}/output1/
mv ${INPUT_DIR}/input1/phi35ragRepo/dataEmbed ${OUTPUT_DIR}/output1/
mv ${INPUT_DIR}/input1/phi35ragRepo/rag ${OUTPUT_DIR}/output1/
