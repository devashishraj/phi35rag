import requests
import json
import hashlib
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone


outputPath=os.getenv("Fetch_OUTPUT")

def hash_to_sha512_string(article):
    """
    Generates a 10-character hash from a given string.

    Args:
        input_string: The string to hash.

    Returns:
        A 10-character hexadecimal hash string, or None if input is not a string.
    """
    if not isinstance(article, str):
        return None  # Handle non-string input

    hash_object = hashlib.sha512(article.encode())
    hex_digest = hash_object.hexdigest()
    return hex_digest[:10]  # Return the first 10 characters

def fetch_wikipedia_articles_by_category(category, limit=10):
    """
    Fetches article titles from a specific Wikipedia category.

    Args:
        category (str): The name of the Wikipedia category.
        limit (int): The maximum number of articles to fetch.

    Returns:
        list: A list of article titles in the category.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    article_titles = []  # List to store article titles
    continue_token = None  # For handling pagination

    while len(article_titles) < limit:
        # Prepare query parameters
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit - len(article_titles), 500),  # Max 500 per request
            "cmtype": "page",
        }

        # Add continuation token if it exists
        if continue_token:
            params["cmcontinue"] = continue_token

        # Send the HTTP GET request
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP {response.status_code}")

        # Parse the JSON response
        data = response.json()

        # Add article titles to the list
        for member in data.get("query", {}).get("categorymembers", []):
            article_titles.append(member["title"])

        # Check if there are more results to fetch
        continue_token = data.get("continue", {}).get("cmcontinue")
        if not continue_token:  # No more pages to fetch
            break

    return article_titles[:limit]


def fetch_article_text(page_title):
    """
    Fetches the full content of a Wikipedia article.

    Args:
        page_title (str): The title of the Wikipedia page.

    Returns:
        str: The content of the article or an empty string if fetching fails.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,  # Fetch plain text, no HTML
        "titles": page_title,
        "format": "json",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(
            f"Error fetching article text for '{page_title}': HTTP {response.status_code}"
        )
        return ""

    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        return page_data.get("extract", "")  # Extract plain text content

    return ""


def format_diff(diff_html):
    """
    Cleans and simplifies the diff content.

    Args:
        diff_html (str): HTML content of the diff.

    Returns:
        str: Simplified and cleaned diff.
    """
    soup = BeautifulSoup(diff_html, "html.parser")

    additions = [ins.get_text() for ins in soup.find_all("ins")]
    deletions = [del_tag.get_text() for del_tag in soup.find_all("del")]

    simplified_diff = []

    if additions:
        simplified_diff.append(f"Added: {', '.join(additions[:])}...") 
    if deletions:
        simplified_diff.append(
            f"Removed: {', '.join(deletions[:])}..."
        ) 

    return "\n".join(simplified_diff) if simplified_diff else None


def get_recent_changes_within_24hrs(page_title, cutoff_time):
    """
    Fetches recent changes made to a Wikipedia page within the last 24 hours.

    Args:
        page_title (str): The title of the Wikipedia page.
        cutoff_time (datetime): Time limit for fetching changes.

    Returns:
        list: A list of changes with metadata and diffs.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "timestamp|comment|ids|diff",
        "rvdiffto": "prev",
        "rvlimit": 10,
        "format": "json",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(
            f"Error fetching revisions for '{page_title}': HTTP {response.status_code}"
        )
        return []

    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    revisions_data = []

    for page_id, page_info in pages.items():
        revisions = page_info.get("revisions", [])
        for rev in revisions:
            timestamp = rev.get("timestamp")
            revision_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            if revision_time < cutoff_time:
                continue

            comment = rev.get("comment", "No comment")
            raw_diff = rev.get("diff", {}).get("*", "No diff available")

            clean_diff = format_diff(raw_diff)
            if clean_diff:
                revisions_data.append(
                    {
                        "change_id": f"{page_id}_{rev.get('revid')}",
                        "timestamp": timestamp,
                        "change_summary": comment,
                        "diff": clean_diff,
                    }
                )

    return revisions_data


def main():
    category = "Programming languages"
    limit = 200
    cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=24)

    dataset = []

    print(f"Fetching up to {limit} articles from the '{category}' category...")
    try:
        articles = fetch_wikipedia_articles_by_category(category, limit)
        print("\nArticles found:")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article}")

        print("\nFetching recent changes (within 24 hours) for these articles...")
        for article in articles:
            changes = get_recent_changes_within_24hrs(article, cutoff_time)
            articleID=hash_to_sha512_string(article)
            # articleID = hash(article)
            # Only add articles with changes to the dataset
            if changes:
                article_text = fetch_article_text(article)  # Fetch the article content

                article_data = {
                    "article_id": articleID,  # Unique identifier based on article title
                    "title": article,
                    "content": {
                        "sections": [
                            {
                                "section_title": "Main Article",
                                "text": article_text
                                if article_text
                                else f"Could not fetch text for '{article}'.",
                                "changes": changes,
                            }
                        ]
                    },
                }
                dataset.append(article_data)

        # Save the dataset as JSON
        if dataset:  # Only save if there are articles with changes
            with open(outputPath, "w") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            print("articles with recent changes saved")
        else:
            print("No recent changes found. Dataset not created.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
