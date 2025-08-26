"""
Unit tests for the download module.
"""

import io
from unittest.mock import patch, MagicMock

import httpx
from PIL import Image

from text_image_similarity.download_image_url import download_image


def test_download_image_success():
    """Tests that download_image returns an image on success."""
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100)).save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.content = img_byte_arr

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response
        image = download_image("http://example.com/image.jpg", 10, 0)
        assert isinstance(image, Image.Image)


def test_download_image_failure():
    """Tests that download_image returns None on network failure."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = (
            httpx.RequestError("mock error")
        )
        image = download_image("http://example.com/image.jpg", 10, 0)
        assert image is None
