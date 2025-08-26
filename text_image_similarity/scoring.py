"""
Public API for the text_image_similarity package.
"""

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, Tuple

import torch

from .config import Config
from .download_image_url import download_image
from .record_class import Record

logger = logging.getLogger(__name__)


def _batched(iterable: Iterable, n: int) -> Iterable[Tuple]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def score_pairs(
    records: Iterable[Record],
    model,
    processor,
    config: Config,
) -> Iterable[float]:
    """
    Computes similarity scores for image-text records using a pre-loaded model.

    This function processes records in batches to manage memory usage and is
    implemented as a generator, yielding scores as they are computed.

    Args:
        records: An iterable of Record objects.
        model: The pre-loaded CLIP model.
        processor: The pre-loaded CLIP processor.
        config: Configuration for the scoring process.

    Yields:
        A float similarity score for each input record, aligned with the input order.
        Yields `float('nan')` for any record that fails processing.
    """
    # Prepare a download function with fixed timeout and retries
    downloader = partial(
        download_image, timeout=config.timeout_s, retries=config.retries
    )

    with ThreadPoolExecutor(max_workers=config.max_io_workers) as executor:
        for batch_records in _batched(records, config.batch_size):
            images = []
            captions = []
            valid_indices_in_batch = []

            urls = [record.url for record in batch_records]
            downloaded_images = executor.map(downloader, urls)

            for i, (record, image) in enumerate(zip(batch_records, downloaded_images)):
                if image:
                    images.append(image)
                    captions.append(record.caption)
                    valid_indices_in_batch.append(i)
                else:
                    logger.warning(f"Failed to download or process image: {record.url}")

            batch_scores = {}  # Maps index_in_batch to its score

            if images:
                try:
                    inputs = processor(
                        text=captions,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(config.device)

                    use_fp16 = config.precision == "fp16" and "cuda" in config.device
                    with torch.inference_mode(), torch.autocast(
                        device_type=config.device.split(":")[0], enabled=use_fp16
                    ):
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        scores = logits_per_image.diag().squeeze().cpu()

                    # Handle case of single-item batch
                    if scores.ndim == 0:
                        scores_list = [scores.item()]
                    else:
                        scores_list = scores.tolist()

                    for i, score in zip(valid_indices_in_batch, scores_list):
                        batch_scores[i] = float(score)

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # On batch failure, all valid items in this batch get NaN
                    for i in valid_indices_in_batch:
                        batch_scores[i] = float("nan")

            # Yield a result for each record in the original batch to maintain order
            for i in range(len(batch_records)):
                yield batch_scores.get(i, float("nan"))
