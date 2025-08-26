"""
Main script to run the text-image similarity scoring process.
"""

import logging
from text_image_similarity.config import Config
from text_image_similarity.pipeline import score_csv


def main():
    """Main entry point for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_csv = "challenge_set.csv"
    output_csv = "challenge_scored.csv"
    config = Config()

    try:
        score_csv(
            in_path=input_csv,
            out_path=output_csv,
            config=config,
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"A critical error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
