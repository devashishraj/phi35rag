# Run the data preparation script

echo "getWikiChangesToJson.py"
python getWikiChangesToJson.py

#move output to sharedDir
echo "moving output to ${OUTPUT_DIR}/step to generate quereis for RAG "

mv wikipedia_article_changes.json ${INPUT_DIR}/model_repo/Step1/generateQueries/

#moving only relevant part to next stage
mv ${INPUT_DIR}/model_repo/step1/generateQueries/ ${OUTPUT_DIR}/step2/
mv ${INPUT_DIR}/model_repo/step1/processData/ ${OUTPUT_DIR}/step2/
mv ${INPUT_DIR}/ragRepo/step1/ragPrompt/ ${OUTPUT_DIR}/step2/
