import json
import re
import os
import random
import sys
from llama_cpp import Llama
import logging
from rich.console import Console
from rich.markdown import Markdown

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console(width=120)

# Paths
llm_path = "/app/Phi-3.5-mini-instruct-Q6_K.gguf"
articles_file_path = "WikiRC_ES.json"
output_file_path = "WikiRC_ESO.json"

# Function to check file existence
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

# Verify required files
check_path(llm_path)
check_path(articles_file_path)

# Load articles
try:
    with open(articles_file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
except json.JSONDecodeError:
    logging.error("Failed to decode JSON. Check file format.")
    sys.exit(1)

if not isinstance(articles, list) or not articles:
    logging.error("Invalid or empty articles data.")
    sys.exit(1)


# Function to clean and format text
def clean_text(text: str) -> str:
    text = re.sub(r"==\s*(References|External links)\s*==.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\[[0-9]+\]", "", text)  # Remove citation numbers
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

# Initialize Model
try:
    model = Llama(model_path=llm_path, 
                  n_gpu_layers=-1,
                  n_threads=6, 
                  n_ctx=14000, 
                  stop=["<|endoftext|>", "<|end|>"]
                  )
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

# Function to generate summaries
def generate_summary(article):
    title = article.get("title", "Unknown Title")
    sections = article.get("content", {}).get("sections", [])
    
    if not sections:
        logging.warning(f"No sections found for article: {title}")
        return "No content available for summarization."
    
    main_text = ""
    recenttChange=""
    for section in sections:
        main_text = f"\nArticle:\n{clean_text(section.get("text", ""))}\n\n"
        changes = section.get("changes", [])
        if changes:
            changesText = "\n".join(f"- {chg.get('change_summary', 'No summary')}" for chg in changes)
            recenttChange = f"\n[Recent Changes]:\n{changesText}\n\n"
    
    prompt = f"""
<|system|>
Your objective is to summarize the following article in structured and accurate manner.
Ensure to capture the main points, themes, covers key aspects, historical context and practical usage. 
If context given mentions recent changes made into it and if them seem meaningful 
and relvent to article incorporate them in your summary. 
Article might have latest infomartion so don't factor in knowledge cutoff date.
Do not use outside knoweldge and never mention anything about given instructions.
<|end|>

<|user|>
{main_text}
{recenttChange}
<|end|>

<|assistant|>
"""

    
    with console.status("[bold green]Generating summary..."):
        output = model.create_completion(
            prompt=prompt,
            max_tokens=4200,
            stop=["<|end|>"],
            frequency_penalty=0.1,
            presence_penalty=0.3,
            temperature=0.4,
        )
    
    response = output["choices"][0]["text"].strip()
    return response

# Process selected articles and store summaries
for article in articles:
    summary = generate_summary(article)
    article["oneShotSummary"] = summary
    console.print(Markdown(f"### Summary for {article['title']}\n{summary}"))
    console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"Summaries saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save summaries: {e}")
