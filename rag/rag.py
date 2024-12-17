import json
import os
import sys
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import logging
from rich.console import Console
from rich.markdown import Markdown

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_file(file_path):
    # Check if file exists
    if os.path.exists(file_path):
        print(f"Model file found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
    else:
        print(f"Model file NOT found: {file_path}")
        print("Please check the file path")
        sys.exit(1)


# Paths
model_path = "/app/Phi-3.5-mini-instruct-Q4_0_4_4.gguf"
vectorstore_path = "vectorstore_index.faiss"
queries_file_path = "ragQueries.json"
output_file_path = "ragResponses.json"

check_file(model_path)
check_file(vectorstore_path)
check_file(queries_file_path)

# Initialize Model
model = Llama(
    model_path=model_path,
    n_gpu_layers=1,  # Adjust based on your GPU capability
    n_threads=8,
    temperature=0.1,
    top_p=0.5,
    n_ctx=8192,
    repeat_penalty=1.4,
    stop=["<|endoftext|>", "<|end|>"],
    verbose=False,
)

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
        if len(context.split()) > model.n_ctx:
            context = " ".join(context.split()[: model.n_ctx])

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


# Main Function to Load Vector Store and Perform RAG
def main():
    try:
        # Load vectorstore
        vectorstore = FAISS.load_local(vectorstore_path)
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
