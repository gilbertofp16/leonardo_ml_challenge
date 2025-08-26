"""
Unit tests for the scoring module.
"""

import pandas as pd
import pytest
import torch
from PIL import Image
from text_image_similarity.config import Config
from text_image_similarity.record_class import Record
from text_image_similarity.result import ScoreResult
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

    def eval(self):
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

    results = list(score_pairs(records, model, processor, config))

    assert len(results) == 2
    assert all(isinstance(r, ScoreResult) for r in results)
    # 25.0 normalized and rounded -> round(0.625, 4) = 0.625
    assert results[0].score == 0.625
    assert results[0].error is None
    assert results[1].score == 0.625
    assert results[1].error is None


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

    results = list(score_pairs(records, model, processor, config))

    assert len(results) == 2
    assert results[0].score == 0.625
    assert results[0].error is None
    assert pd.isna(results[1].score)
    assert results[1].error == "Image download failed"
