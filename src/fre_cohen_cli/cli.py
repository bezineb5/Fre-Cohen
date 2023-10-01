"""Minimal CLI for fre_cohen package."""

import argparse
import logging

from fre_cohen import configuration
from fre_cohen.ingestion import CSVIngestion
from fre_cohen.semantic_layer import OpenAISemanticInterpretation

logger = logging.getLogger(__name__)


def _parse_arguments():
    """Parses the arguments"""

    parser = argparse.ArgumentParser(description="CLI for fre_cohen package")
    parser.add_argument(
        "--input",
        type=str,
        help="The input CSV file",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The output file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    # OpenAI API key
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="The OpenAI API key",
    )
    return parser.parse_args()


def main():
    """Main function"""

    args = _parse_arguments()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    open_ai_key = args.openai_api_key
    config = configuration.Config(openai_api_key=open_ai_key)

    # Ingest data from the CSV
    ingestion = CSVIngestion(path=args.input)
    # data = ingestion.get_data()
    metadata = ingestion.get_metadata()

    # Add semantic information
    sem_interpretation = OpenAISemanticInterpretation(config=config, fields=metadata)
    fields_graph = sem_interpretation.get_data_structure()

    logger.info("Semantic information:")
    logger.info(fields_graph)


if __name__ == "__main__":
    main()
