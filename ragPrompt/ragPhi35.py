# Imports
from rich.markdown import Markdown
from rich.console import Console
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from llama_cpp import Llama
import json
import warnings
from langchain.schema import Document
import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Temporary solution for multiple libomp.dylib presence
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings(action="ignore")

console = Console(width=90)

# Paths
embpath = "all-MiniLM-L6-v2.F16.gguf"
model_path = "Phi-3.5-mini-instruct-Q4_0_4_4.gguf"
json_file_path = "wikipedia_article_changes.json"

# Initialize Embeddings
embeddings = LlamaCppEmbeddings(model_path=embpath)

# Initialize Model
model = Llama(
    model_path=model_path,
    n_gpu_layers=1,  # Adjust based on your GPU capability
    n_threads=8,  # Reduce threads for lower CPU usage
    temperature=0.1,
    top_p=0.5,
    n_ctx=8192,  # Reduce context size for stability
    repeat_penalty=1.4,
    stop=["<|endoftext|>", "<|end|>"],  # Adjusted for the model's requirements
    verbose=False,  # Disable verbose logging
)


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


# Load and preprocess JSON
def load_and_process_json(json_file_path):
    try:
        documents = parse_json(json_file_path)
        return preprocess_documents(documents)
    except Exception as e:
        logging.error(f"Failed to load and process JSON: {e}")
        return []


processed_docs = load_and_process_json(json_file_path)

# Split documents into chunks
text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(processed_docs)

# Create vectorstore
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


# Q&A Function
def PhiQnA(question, retriever, hits, maxtokens, model):
    """
    Perform Q&A based on the context retrieved from the vectorstore.

    Args:
        question (str): The user's question.
        retriever: The initialized retriever object.
        hits (int): Number of documents to retrieve.
        maxtokens (int): Maximum tokens for the response.
        model: The initialized language model.

    Returns:
        tuple: Response string and the retrieved documents.
    """
    try:
        docs = retriever.invoke(question)

        # Combine content into a single context string
        context = "\n".join(doc.page_content for doc in docs)
        template = f"""<|system|>
You are a Language Model trained to answer questions based on the provided context.
<|end|>

<|user|>
Answer the question based only on the following context:

[Context]
{context}
[End of Context]

Question: {question}
<|end|>
<|assistant|>
"""
        # Generate response
        with console.status("[bold green]Generating response..."):
            output = model.create_completion(
                prompt=template,
                max_tokens=maxtokens,
                stop=["<|end|>"],
                temperature=0.1,
            )

        response = output["choices"][0]["text"].strip()
        return response, docs

    except Exception as e:
        logging.error(f"Error during Q&A generation: {e}")
        return "An error occurred while generating the response.", []


# Example Query
if __name__ == "__main__":
    question = "Give me a summary based on recent changes in Concurrent Haskell"
    try:
        response, retrieved_docs = PhiQnA(
            question, retriever, hits=3, maxtokens=500, model=model
        )

        # Print Results
        console.print(Markdown(f"### Question:\n{question}"))
        console.print(Markdown(f"### Response:\n{response}"))
    except Exception as e:
        logging.error(f"Error during execution: {e}")
