"""
Image downloading utilities.
"""

import io
import logging
from typing import Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


def download_image(url: str, timeout: int, retries: int) -> Optional[Image.Image]:
    """
    Downloads an image from a URL with retries and timeout.

    Args:
        url: The URL of the image to download.
        timeout: The timeout in seconds for the request.
        retries: The number of retries to attempt.

    Returns:
        A PIL Image object on success, or None on failure.
    """
    transport = httpx.HTTPTransport(retries=retries)
    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            # Try to open the image to ensure it's valid
            image = Image.open(io.BytesIO(response.content))
            # Verify image data can be loaded
            image.load()
            return image
    except httpx.RequestError as e:
        logger.warning(f"Network error downloading {url}: {e}")
        return None
    except Image.UnidentifiedImageError:
        logger.warning(f"Could not identify image file from {url}.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {url}: {e}")
        return None
