"""
tests/test_feature_cache.py

Unit tests cho module FeatureCache và CachedAudioDataset (W3-02):
- Kiểm tra tính hợp lệ của FeatureCache
- Kiểm tra CachedAudioDataset với dữ liệu giả lập và dữ liệu thực tế
- Kiểm tra tính tương thích với PyTorch DataLoader
- Kiểm tra hook biến đổi (transform / augmentation)
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_cache import FeatureCache, CachedAudioDataset


def test_cached_dataset_synthetic():
    """Kiểm tra hoạt động của CachedAudioDataset với mảng numpy giả lập."""
    num_samples = 50
    X_synth = np.random.rand(num_samples, 1, 64, 63).astype(np.float32)
    y_synth = np.random.randint(0, 9, size=num_samples, dtype=np.int64)

    dataset = CachedAudioDataset(X_synth, y_synth)
    assert len(dataset) == num_samples

    # Kiểm tra lấy 1 mẫu
    feat, label = dataset[0]
    assert isinstance(feat, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert feat.shape == (1, 64, 63)
    assert feat.dtype == torch.float32
    assert label.dtype == torch.int64


def test_dataloader_compatibility():
    """Kiểm tra tương thích với DataLoader khi batching và shuffle."""
    num_samples = 64
    batch_size = 16
    X_synth = np.random.rand(num_samples, 1, 64, 63).astype(np.float32)
    y_synth = np.random.randint(0, 9, size=num_samples, dtype=np.int64)

    dataset = CachedAudioDataset(X_synth, y_synth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_seen = 0
    for x_batch, y_batch in loader:
        assert x_batch.shape == (batch_size, 1, 64, 63)
        assert y_batch.shape == (batch_size,)
        total_seen += len(x_batch)

    assert total_seen == num_samples


def test_transform_hook():
    """Kiểm tra transform hook được áp dụng chính xác cho từng tensor."""
    X_synth = np.ones((5, 1, 64, 63), dtype=np.float32)
    y_synth = np.zeros(5, dtype=np.int64)

    # Transform nhân đôi giá trị tensor
    def double_transform(tensor):
        return tensor * 2.0

    dataset = CachedAudioDataset(X_synth, y_synth, transform=double_transform)
    feat, _ = dataset[0]
    assert torch.allclose(feat, torch.full((1, 64, 63), 2.0))


def test_feature_cache_real_data():
    """Kiểm tra nạp dữ liệu thật nếu cache đã được trích xuất."""
    cache = FeatureCache()
    if not cache.is_cached("train"):
        pytest.skip("Chưa tạo cache data/processed/features, bỏ qua test dữ liệu thật.")

    X_train, y_train, _ = cache.load_split("train")
    assert X_train.shape[1:] == (1, 64, 63)
    assert len(X_train) == len(y_train)
    assert 0.0 <= X_train.min() and X_train.max() <= 1.0001

    dataset = cache.get_dataset("train")
    assert len(dataset) == len(X_train)
