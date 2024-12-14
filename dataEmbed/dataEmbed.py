# save_vectorstore.py

from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from llama_cpp import Llama
from langchain.schema import Document
import json
import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
embpath = "/app/all-MiniLM-L6-v2.F16.gguf"
json_file_path = "wikipedia_article_changes.json"
vectorstore_path = "vectorstore_index.faiss"

# Initialize Embeddings
embeddings = LlamaCppEmbeddings(model_path=embpath)


# Function to parse JSON input
def parse_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON file.")
        raise

    documents = []
    try:
        for article in data:
            article_id = article["article_id"]
            title = article["title"]
            sections = article["content"]["sections"]

            for section in sections:
                section_title = section["section_title"]
                text = section.get("text", "")
                for change in section.get("changes", []):
                    change_summary = change.get("change_summary", "")
                    diff = change.get("diff", "")
                    documents.append(
                        {
                            "article_id": article_id,
                            "title": title,
                            "section_title": section_title,
                            "text": text,
                            "change_summary": change_summary,
                            "diff": diff,
                        }
                    )
    except KeyError as e:
        logging.error(f"Missing expected key in JSON structure: {e}")
        raise

    return documents


# Preprocess documents into langchain Document format
def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        content = (
            f"Article Title: {doc['title']}\n"
            f"Section Title: {doc['section_title']}\n"
            f"Change Summary: {doc['change_summary']}\n"
            f"Diff: {doc['diff']}\n\n"
            f"Full Text: {doc['text']}\n"
        )
        processed_docs.append(Document(page_content=content, metadata={}))
    return processed_docs


# Main Function to Create and Save Vector Store
def main():
    documents = preprocess_documents(parse_json(json_file_path))
    text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create and save vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vectorstore_path)
    logging.info(f"Vector store saved to {vectorstore_path}")


if __name__ == "__main__":
    main()
