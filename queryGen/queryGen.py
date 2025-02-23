import json
import random
import logging
import sys
import os

input_file="WikiRC.json"
output_file="WikiRC_Q.json"

def setup_logging(level=logging.INFO):
    """
    Configure logging to stderr with a dynamic logging level.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("wiki_query_generator")
    logger.setLevel(level)

    # Stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    return logger


def generate_queries_from_wikipedia_changes(
    max_queries=3,
    # Fetch the query template
    query_template=os.getenv(
        "QUERY_TEMPLATE",
        "Summarize {title} Wikipedia article based around recent changes in the article",
    ),
):
    """
    Generate queries about recent Wikipedia article changes with comprehensive logging.

    Args:
        input_file (str): Path to the input JSON file with Wikipedia changes
        output_file (str): Path to save the output query JSON file
        max_queries (int): Maximum number of queries to generate
        query_template (str): Template for generating queries

    Returns:
        bool: True if successful, False otherwise
    """
    # Setup logging
    logger = setup_logging()

    try:
        # Log start of process
        logger.info(f"Starting query generation process")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Max queries: {max_queries}")

        # Validate input file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                articles_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Input file '{input_file}' not found")
            return False
        except json.JSONDecodeError:
            logger.error(f"Unable to parse JSON from '{input_file}'")
            return False

        # Validate data
        if not isinstance(articles_data, list):
            logger.error("Input JSON is not a list")
            return False

        invalid_articles = [
            article for article in articles_data if "title" not in article
        ]
        if invalid_articles:
            logger.warning(f"Some articles are missing 'title': {invalid_articles}")
            articles_data = [article for article in articles_data if "title" in article]

        if not articles_data:
            logger.error("No valid articles with 'title' found in input JSON")
            return False

        # Randomly select unique articles
        try:
            selected_articles = random.sample(
                articles_data, min(max_queries, len(articles_data))
            )
        except ValueError as e:
            logger.error(f"Error selecting articles: {e}")
            return False

        # Generate queries
        queries = []
        for article in selected_articles:
            query = {
                "article_title": article["title"],
                "query": query_template.format(title=article["title"]),
            }
            queries.append(query)

        # Ensure output file exists or create it
        try:
            if not os.path.exists(output_file):
                logger.info(f"Output file '{output_file}' does not exist. Creating it.")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump([], f)  # Create an empty JSON array

            # Save queries to the output JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(queries, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logger.error(f"Error writing to output file: {e}", exc_info=True)
            return False

        # Log success details
        logger.info(f"Successfully generated {len(queries)} queries")
        logger.info("Generated queries:")
        for q in queries:
            logger.info(f"- {q['query']}")

        return True

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False
    finally:
        logger.info("Query generation process completed")


def main():
    # Run the query generation with logging
    success = generate_queries_from_wikipedia_changes()

    # Log and set exit code based on success
    if success:
        logging.info("Script completed successfully. Exiting with code 0.")
        sys.exit(0)
    else:
        logging.error("Script encountered errors. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
