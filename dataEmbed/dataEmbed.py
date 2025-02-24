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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_path(path):
    if not os.path.exists(path):
        logging.error(f"Path not found: {path}")
        sys.exit(1)
    if not os.access(path, os.R_OK):
        logging.error(f"Path is not readable: {path}")
        sys.exit(1)
    logging.info(f"Path verified: {path}")

# Paths
embpath = "/app/jinv3"
modelCachePath = "/app/jinv3/modelCache"
json_file_path = "WikiRC_Q.json"
check_path(embpath)
check_path(json_file_path)

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
emb_model_kwargs = {"device": deviceDetected, "local_files_only": True, "trust_remote_code": True}

saveVectorStoreTo = "app/vectorstore_index.faiss"

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name=embpath, model_kwargs=emb_model_kwargs, cache_folder=modelCachePath)

# Proper JSON Parsing
def parse_json(file_path):
    with open(file_path, "r") as file:
        try:
            data = json.load(file)  # Load full JSON array
        except json.JSONDecodeError:
            logging.error("Invalid JSON file format.")
            sys.exit(1)
        
        for article in data:
            article_id = article.get("article_id", "")
            title = article.get("title", "")
            
            for section in article.get("content", {}).get("sections", []):
                section_title = section.get("section_title", "")
                text = section.get("text", "")
                
                for change in section.get("changes", []):
                    change_summary = change.get("change_summary", "")
                    diff = change.get("diff", "")
                    
                    yield Document(
                        page_content=f"Article Title: {title}\nArticleID: {article_id}\nSection Title: {section_title}\nChange Summary: {change_summary}\nDiff: {diff}\n\nFull Text: {text}\n",
                        metadata={"articleID": str(article_id)},
                    )

# Optimized Text Splitter
def split_documents(documents):
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=20)  # Reduced overlap
    for doc in documents:
        yield from text_splitter.split_documents([doc])

# Process and Index in Batches
def process_and_index():
    vectorstore = None
    batch_size = 1000  # Process 1000 documents at a time
    batch = []
    
    for doc in parse_json(json_file_path):
        batch.append(doc)
        if len(batch) >= batch_size:
            logging.info(f"Processing batch of {len(batch)} documents")
            texts = list(split_documents(batch))
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(texts, embeddings)
            else:
                vectorstore.add_documents(texts)
            batch.clear()
    
    if batch:  # Process any remaining documents
        logging.info(f"Processing final batch of {len(batch)} documents")
        texts = list(split_documents(batch))
        if vectorstore is None:
            vectorstore = FAISS.from_documents(texts, embeddings)
        else:
            vectorstore.add_documents(texts)
    
    if vectorstore:
        vectorstore.save_local(saveVectorStoreTo)
        logging.info(f"Vector store saved to {saveVectorStoreTo}")

def main():
    process_and_index()

if __name__ == "__main__":
    main()

