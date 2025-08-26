"""
Integration test for the CSV pipeline.
"""

import pandas as pd
import pytest
from PIL import Image
from text_image_similarity.config import Config
from text_image_similarity.pipeline import score_csv


@pytest.fixture
def mock_downloader_integration(monkeypatch):
    """Mocks the image downloader to return a local dummy image."""
    dummy_image = Image.new("RGB", (100, 100), color="blue")
    monkeypatch.setattr(
        "text_image_similarity.scoring.download_image",
        lambda url, timeout, retries: dummy_image,
    )


@pytest.fixture
def setup_csv_fixture(tmp_path):
    """Creates a dummy CSV file for testing."""
    csv_path = tmp_path / "input.csv"
    data = {
        "url": ["http://example.com/image1.jpg"],
        "caption": ["A test caption"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_csv_pipeline_integration(
    mock_downloader_integration, setup_csv_fixture, tmp_path
):
    """
    Tests the full CSV pipeline from input to output.
    """
    input_csv = setup_csv_fixture
    output_csv = tmp_path / "output.csv"
    config = Config()

    score_csv(
        in_path=str(input_csv),
        out_path=str(output_csv),
        config=config,
    )

    assert output_csv.exists()

    df = pd.read_csv(output_csv)

    assert "similarity" in df.columns
    assert "error" in df.columns
    assert len(df) == 1
    assert not pd.isna(df["similarity"].iloc[0])
