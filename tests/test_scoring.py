"""
Unit tests for the scoring module.
"""

import pandas as pd
import pytest
import torch
from PIL import Image
from text_image_similarity.config import Config
from text_image_similarity.record_class import Record
from text_image_similarity.scoring import score_pairs


class MockCLIPModel:
    """A mock CLIP model for testing."""

    def __call__(self, **kwargs):
        # Return a dummy output with logits
        batch_size = kwargs.get("input_ids", kwargs.get("pixel_values")).shape[0]
        return type(
            "obj",
            (object,),
            {"logits_per_image": torch.diag(torch.ones(batch_size) * 25.0)},
        )()

    def to(self, device):
        return self


class MockBatchEncoding:
    """Mocks the BatchEncoding object returned by the processor."""

    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self.data


class MockCLIPProcessor:
    """A mock CLIP processor for testing."""

    def __call__(self, text, images, return_tensors, padding, truncation):
        data = {
            "input_ids": torch.ones((len(text), 77)),
            "pixel_values": torch.ones((len(images), 3, 224, 224)),
        }
        return MockBatchEncoding(data)


@pytest.fixture
def mock_downloader(monkeypatch):
    """Mocks the image downloader."""

    def mock_download(url, timeout, retries):
        if "fail" in url:
            return None
        return Image.new("RGB", (100, 100), color="red")

    monkeypatch.setattr("text_image_similarity.scoring.download_image", mock_download)


def test_score_pairs_returns_correct_scores(mock_downloader):
    """
    Tests that score_pairs returns a score for each valid record.
    """
    records = [
        Record(url="http://example.com/image1.jpg", caption="A test image"),
        Record(url="http://example.com/image2.jpg", caption="Another test image"),
    ]
    config = Config(device="cpu")
    model, processor = MockCLIPModel(), MockCLIPProcessor()

    scores = list(score_pairs(records, model, processor, config))

    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    assert all(s == 25.0 for s in scores)


def test_score_pairs_handles_download_failure(mock_downloader):
    """
    Tests that score_pairs returns NaN for records that fail to download.
    """
    records = [
        Record(url="http://example.com/image1.jpg", caption="A test image"),
        Record(url="http://example.com/fail_image.jpg", caption="This one will fail"),
    ]
    config = Config(device="cpu")
    model, processor = MockCLIPModel(), MockCLIPProcessor()

    scores = list(score_pairs(records, model, processor, config))

    assert len(scores) == 2
    assert scores[0] == 25.0
    assert pd.isna(scores[1])
