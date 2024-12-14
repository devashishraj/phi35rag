# load_vectorstore_rag.py

from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import logging
from rich.console import Console
from rich.markdown import Markdown

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
model_path = "Phi-3.5-mini-instruct-Q4_0_4_4.gguf"
vectorstore_path = "vectorstore_index.faiss"

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


# Main Function to Load Vector Store and Perform RAG
def main():
    vectorstore = FAISS.load_local(vectorstore_path)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

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


if __name__ == "__main__":
    main()
