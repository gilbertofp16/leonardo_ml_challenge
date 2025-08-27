"""
CSV processing pipeline for scoring image-text pairs.
"""

import logging
import pandas as pd
import torch

from .config import Config
from .record_class import Record
from .model_loader import get_model_and_processor
from .scoring import score_pairs

logger = logging.getLogger(__name__)


def score_csv(
    in_path: str,
    out_path: str,
    config: Config,
    col_url: str = "url",
    col_text: str = "caption",
) -> None:
    """
    Reads a CSV file, computes similarity scores, and writes to a new CSV.

    This function processes the CSV in chunks to handle large files efficiently.

    Args:
        in_path: Path to the input CSV file.
        out_path: Path to write the output CSV file.
        config: Configuration for the scoring process.
        col_url: The name of the column containing image URLs.
        col_text: The name of the column containing captions.
    """
    logger.info(f"Starting CSV scoring process from '{in_path}' to '{out_path}'.")

    model, processor = get_model_and_processor()

    # Resolve device
    device = config.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Device 'auto' resolved to '{device}'.")

    model.to(device)
    model.eval()

    # Use a generator to read the CSV in chunks
    reader = pd.read_csv(in_path, chunksize=config.batch_size)

    header = True  # Write header only for the first chunk
    for chunk in reader:
        records = (
            Record(url=row[col_url], caption=row[col_text])
            for _, row in chunk.iterrows()
        )

        # Create a new config with the resolved device to pass down
        updated_config = Config(
            batch_size=config.batch_size,
            device=device,
            precision=config.precision,
            timeout_s=config.timeout_s,
            retries=config.retries,
            chunksize=config.chunksize,
            max_io_workers=config.max_io_workers,
        )
        results = list(score_pairs(records, model, processor, updated_config))

        scores, errors = zip(*[(res.score, res.error or "") for res in results])
        chunk["similarity"] = scores
        chunk["error"] = errors

        mode = "w" if header else "a"
        chunk.to_csv(out_path, mode=mode, header=header, index=False)
        header = False

    logger.info("CSV scoring process completed successfully.")
