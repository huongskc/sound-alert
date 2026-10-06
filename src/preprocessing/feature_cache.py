"""
src/preprocessing/feature_cache.py

Module quản lý và nạp bộ nhớ đệm đặc trưng Log-Mel Spectrogram:
- FeatureCache: Nạp dữ liệu đặc trưng mảng NumPy (hỗ trợ in-memory RAM hoặc memory-mapped disk).
- CachedAudioDataset: Dataset wrapper tương thích với PyTorch DataLoader, hỗ trợ data augmentation.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features"


class FeatureCache:
    """
    Quản lý bộ nhớ đệm đặc trưng âm thanh cho các tập train, val, test.
    """

    def __init__(self, cache_dir: Optional[Path] = None, in_memory: bool = True):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.in_memory = in_memory
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def is_cached(self, split: str) -> bool:
        """Kiểm tra xem tập split đã có sẵn cache tổng hợp trên đĩa hay chưa."""
        x_path = self.cache_dir / f"X_{split}.npy"
        y_path = self.cache_dir / f"y_{split}.npy"
        return x_path.exists() and y_path.exists()

    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Nạp dữ liệu đặc trưng và nhãn của tập split.
        Trả về: (X, y, y_onehot)
        X shape: [N, 1, 64, 63] (float32)
        y shape: [N] (int64)
        y_onehot shape: [N, 9] (float32)
        """
        if split in self._cache:
            return self._cache[split]

        x_path = self.cache_dir / f"X_{split}.npy"
        y_path = self.cache_dir / f"y_{split}.npy"
        y_onehot_path = self.cache_dir / f"y_{split}_onehot.npy"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file cache cho tập '{split}' tại {self.cache_dir}.\n"
                f"Vui lòng chạy 'python scripts/extract_features.py' trước!"
            )

        if self.in_memory:
            x_arr = np.load(str(x_path))
            y_arr = np.load(str(y_path))
            y_onehot_arr = np.load(str(y_onehot_path)) if y_onehot_path.exists() else None
        else:
            # Memory-mapped mode
            x_arr = np.load(str(x_path), mmap_mode="r")
            y_arr = np.load(str(y_path), mmap_mode="r")
            y_onehot_arr = np.load(str(y_onehot_path), mmap_mode="r") if y_onehot_path.exists() else None

        if self.in_memory:
            self._cache[split] = (x_arr, y_arr, y_onehot_arr)

        return x_arr, y_arr, y_onehot_arr

    def get_dataset(self, split: str, transform: Optional[Callable] = None) -> "CachedAudioDataset":
        """Tạo PyTorch Dataset từ bộ nhớ đệm."""
        X, y, y_onehot = self.load_split(split)
        return CachedAudioDataset(X, y, y_onehot, transform=transform)

    def benchmark_throughput(self, split: str = "train", batch_size: int = 32, num_workers: int = 0) -> Dict[str, Any]:
        """Đo đạc tốc độ cấp phát dữ liệu (samples/giây) và độ trễ nạp dữ liệu."""
        dataset = self.get_dataset(split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        t0 = time.perf_counter()
        total_samples = 0
        batches = 0

        for x_batch, y_batch in loader:
            total_samples += len(x_batch)
            batches += 1

        elapsed = time.perf_counter() - t0
        throughput = total_samples / elapsed if elapsed > 0 else 0
        ram_mb = (dataset.X.nbytes + dataset.y.nbytes) / (1024 * 1024)

        return {
            "split": split,
            "total_samples": total_samples,
            "batches": batches,
            "elapsed_seconds": elapsed,
            "samples_per_sec": throughput,
            "ram_consumption_mb": ram_mb
        }


class CachedAudioDataset(Dataset):
    """
    PyTorch Dataset nạp từ bộ nhớ đệm NumPy mảng sẵn, tương thích tối đa với DataLoader.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_onehot: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None
    ):
        self.X = X
        self.y = y
        self.y_onehot = y_onehot
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.X[idx]  # [1, 64, 63]
        target = self.y[idx]

        # Chuyển đổi sang PyTorch Tensor (đảm bảo copy writable an toàn)
        feat_tensor = torch.from_numpy(np.array(feat, copy=True)).float()
        target_tensor = torch.tensor(target, dtype=torch.long)

        # Áp dụng data augmentation nếu có (SpecAugment, Noise)
        if self.transform is not None:
            feat_tensor = self.transform(feat_tensor)

        return feat_tensor, target_tensor
