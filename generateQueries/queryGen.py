import json
import random
import logging
import sys


def setup_logging():
    """
    Configure logging to stderr.

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("wiki_query_generator")
    logger.setLevel(logging.INFO)

    # Stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    return logger


def generate_queries_from_wikipedia_changes(
    input_file="wikipedia_article_changes.json",
    output_file="article_queries.json",
    max_queries=3,
):
    """
    Generate queries about recent Wikipedia article changes with comprehensive logging.

    Args:
        input_file (str): Path to the input JSON file with Wikipedia changes
        output_file (str): Path to save the output query JSON file
        max_queries (int): Maximum number of queries to generate

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
        if not articles_data:
            logger.warning("No articles found in the input file")
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
                "query": f"Give me summary of {article['title']} wiki article based on recent changes in it",
            }
            queries.append(query)

        # Save queries to a new JSON file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(queries, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logger.error(f"Error writing to output file: {e}")
            return False

        # Log success details
        logger.info(f"Successfully generated {len(queries)} queries")
        logger.info("Generated queries:")
        for q in queries:
            logger.info(f"- {q['query']}")

        return True

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        logger.info("Query generation process completed")


def main():
    # Run the query generation with logging
    success = generate_queries_from_wikipedia_changes()

    # Set exit code based on success
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
