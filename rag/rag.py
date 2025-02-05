import json
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from llama_cpp import Llama
import logging
from rich.console import Console
from rich.markdown import Markdown

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Model file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(file_path):
        print(f"Error: Path exists but is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)

    if not os.access(file_path, os.R_OK):
        print(f"Error: Model file is not readable: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model file verified successfully: {file_path}")


# Paths
llm_path = "/app/Phi-3.5-mini-instruct-Q6_K.gguf"
embpath = "/app/all-MiniLM-L6-v2-ggml-model-f16.gguf"
vectorstore_path = (
    "vectorstore_index.faiss"  # .faiss is not a not a file so don't check this
)
queries_file_path = "ragQueries.json"
output_file_path = "ragResponses.json"

check_file(llm_path)
check_file(queries_file_path)
check_file(embpath)

# Initialize Model
model = Llama(model_path=llm_path, 
                  n_gpu_layers=-1,
                  n_threads=8, 
                  n_ctx=12000, 
                  verbose=True
                  )

embeddings = LlamaCppEmbeddings(model_path=embpath)


console = Console(width=90)


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
        if not docs:
            return "No relevant documents found.", []

        # Combine content into a single context string
        context = "\n".join(doc.page_content for doc in docs)

        # Check and truncate context length based on n_ctx
        max_ctx_length = model.n_ctx if isinstance(model.n_ctx, int) else model.n_ctx()
        if len(context.split()) > max_ctx_length:
            context = " ".join(context.split()[:max_ctx_length])

        prompt = f"""
<|system|>
Your objective is to provide a well-structured, concise summary of the Wikipedia article.
Consider historical context, significance, and key aspects. If recent changes are meaningful,
incorporate them into your summary.
<|end|>

<|user|>
Answer the question based only on the following context:
{context}
Question: {question}
<|end|>

<|assistant|>
"""
        # Generate response
        with console.status("[bold green]Generating response..."):
            output = model.create_completion(
            prompt=prompt,
            max_tokens=4200,
            stop=["<|end|>"],
            frequency_penalty=0.1,
            presence_penalty=0.3,
            temperature=0.4,
        )

        response = output["choices"][0]["text"].strip()
        return response, docs

    except Exception as e:
        logging.error(f"Error during Q&A generation: {e}")
        return "An error occurred while generating the response.", []


# Main Function to Load Vector Store and Perform RAG
def main():
    try:
        # Load vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        # Load queries from JSON file
        with open(queries_file_path, "r") as file:
            queries = json.load(file)

        if not queries:
            logging.error("The queries file is empty.")
            return

        results = []

        for query_obj in queries:
            question = query_obj.get("query")
            if not question:
                logging.warning("Skipping a query without a 'query' field.")
                continue

            console.print(Markdown(f"### Question:\n{question}"))

            # Perform Q&A
            response, retrieved_docs = PhiQnA(
                question, retriever, hits=3, maxtokens=500, model=model
            )

            # Save the result
            results.append({"query": question, "response": response})

            # Print Results
            console.print(Markdown(f"### Response:\n{response}"))
            console.print("\n" + "=" * 50 + "\n")

        # Write results to JSON file
        with open(output_file_path, "w") as outfile:
            json.dump(results, outfile, indent=4)

        logging.info(f"Responses saved to {output_file_path}")

    except Exception as e:
        logging.error(f"Error during execution: {e}")


if __name__ == "__main__":
    main()
