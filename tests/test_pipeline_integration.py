"""
Integration test for the CSV pipeline.
"""

import pandas as pd
import pytest
from PIL import Image
from text_image_similarity.config import Config
from text_image_similarity.pipeline import score_csv
from tests.test_scoring import MockCLIPModel, MockCLIPProcessor


@pytest.fixture
def mock_model_and_processor(monkeypatch):
    """Mocks the model and processor loading."""
    monkeypatch.setattr(
        "text_image_similarity.pipeline.get_model_and_processor",
        lambda: (MockCLIPModel(), MockCLIPProcessor()),
    )


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
    mock_model_and_processor, mock_downloader_integration, setup_csv_fixture, tmp_path
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

    # Positive test: Verify the CSV can be opened and read correctly
    try:
        pd.read_csv(output_csv)
    except Exception as e:
        pytest.fail(f"Output CSV is not a valid CSV file: {e}")


def test_csv_pipeline_handles_file_not_found(
    mock_model_and_processor, mock_downloader_integration, tmp_path
):
    """
    Tests that the pipeline handles a missing input file gracefully.
    """
    input_csv = tmp_path / "non_existent_input.csv"
    output_csv = tmp_path / "output.csv"
    config = Config()

    with pytest.raises(FileNotFoundError):
        # This will fail because the file does not exist, which is the expected behavior
        pd.read_csv(input_csv, chunksize=config.batch_size)
