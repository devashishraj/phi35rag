# save_vectorstore.py

from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
import json
import os
import logging
import sys


# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_path(path):
    if not os.path.exists(path):
        print(f"Error: Path not found at {path}", file=sys.stderr)
        sys.exit(1)

    if not (os.path.isfile(path) or os.path.isdir(path)):
        print(f"Error: Path exists but is neither a file nor a directory: {path}", file=sys.stderr)
        sys.exit(1)

    if not os.access(path, os.R_OK):
        print(f"Error: Path is not readable: {path}", file=sys.stderr)
        sys.exit(1)

    path_type = "File" if os.path.isfile(path) else "Directory"
    print(f"{path_type} verified successfully: {path}")


# Paths
embpath = "/app/jinv3"
modelCachePath="/app/jinv3/modelCache"
rawData_path = "WikiRC_Q.json"
vectorstore_path = "vectorstore_index.faiss"


check_path(embpath)
check_path(rawData_path)

CONST_cuda="cuda"
CONST_mps="mps"

def set_device():
    if torch.cuda.is_available():
        device = torch.device(CONST_cuda)
        print("CUDA available")
    elif torch.backends.mps.is_available():
        device = torch.device(CONST_mps)
        print("MPS available")
    else:
        device = torch.device("cpu")
        print("Only CPU available")
    return device

# Set the device
deviceDetected = set_device()
# NOTE:change device to CPU in case of ram <=16GB.
emb_model_kwargs = {"device": deviceDetected,
                        "local_files_only":True,
                        "trust_remote_code":True,
                    } 

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name=embpath,model_kwargs=emb_model_kwargs,cache_folder=modelCachePath)
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
                            "title": title,
                            "section_title": section_title,
                            "article_id": article_id,
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
        articleID = str(doc['article_id'])
        content = (
            f"Article Title: {doc['title']}\n"
            f"ArticleID:{doc['article_id']}\n"
            f"Section Title: {doc['section_title']}\n"
            f"Change Summary: {doc['change_summary']}\n"
            f"Diff: {doc['diff']}\n\n"
            f"Full Text: {doc['text']}\n"
        )
        processed_docs.append(Document(page_content=content, metadata={"articleID":articleID}))
    return processed_docs


# Main Function to Create and Save Vector Store
def main():
    documents = preprocess_documents(parse_json(rawData_path))
    text_splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create and save vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vectorstore_path)
    logging.info(f"Vector store saved to {vectorstore_path}")


if __name__ == "__main__":
    main()
