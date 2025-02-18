"""Tests for data loading functionality."""
import pytest
from pathlib import Path

from yolo_sam_training.data import DataLoader
from yolo_sam_training.utils import Sample

@pytest.fixture
def sample_paths():
    data_dir = Path("data/example_training_data")
    return {
        "image": data_dir / "images" / "87.png",
        "mask": data_dir / "masks" / "87.png",
        "bbox": data_dir / "boxes" / "87.txt"
    }

def test_data_loader_initialization(sample_paths):
    loader = DataLoader(
        image_path=sample_paths["image"],
        mask_path=sample_paths["mask"],
        bbox_path=sample_paths["bbox"]
    )
    assert isinstance(loader, DataLoader)

def test_load_sample(sample_paths):
    loader = DataLoader(
        image_path=sample_paths["image"],
        mask_path=sample_paths["mask"],
        bbox_path=sample_paths["bbox"]
    )
    sample = loader.load_sample()
    assert isinstance(sample, Sample)
    assert sample.image is not None
    assert sample.mask is not None
    assert len(sample.bboxes) > 0 