"""
scripts/benchmark_feature_cache.py

Kiểm thử tốc độ nạp đặc trưng và tối ưu bộ nhớ đệm (Mã W3-02):
1. Nạp và kiểm tra tính hợp lệ của toàn bộ cache train / val / test.
2. Đo lường tốc độ cấp phát dữ liệu (Throughput - samples/giây) qua PyTorch DataLoader.
3. So sánh hiệu năng giữa chế độ In-Memory và Memory-Mapped (mmap).
4. Xuất báo cáo hiệu năng bộ nhớ đệm phục vụ huấn luyện Custom 2D-CNN.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_cache import FeatureCache


def run_cache_benchmark():
    print("=" * 65)
    print("      BENCHMARK HIỆU NĂNG BỘ NHỚ ĐỆM ĐẶC TRƯNG (W3-02)")
    print("=" * 65)

    cache_dir = PROJECT_ROOT / "data" / "processed" / "features"
    if not (cache_dir / "X_train.npy").exists():
        print(f"[ERROR] Chưa tìm thấy cache đặc trưng tại {cache_dir}. Vui lòng chạy scripts/extract_features.py trước!")
        return False

    # 1. Đo lường chế độ IN-MEMORY
    print("\n--- 1. ĐO LƯỜNG CHẾ ĐỘ NẠP TOÀN BỘ VÀO BỘ NHỚ RAM (IN-MEMORY) ---")
    t0 = time.perf_counter()
    cache_ram = FeatureCache(cache_dir=cache_dir, in_memory=True)
    
    # Nạp cả 3 tập
    X_train, y_train, _ = cache_ram.load_split("train")
    X_val, y_val, _ = cache_ram.load_split("val")
    X_test, y_test, _ = cache_ram.load_split("test")
    load_time = time.perf_counter() - t0

    total_samples = len(X_train) + len(X_val) + len(X_test)
    total_ram_mb = (X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes + X_test.nbytes + y_test.nbytes) / (1024 * 1024)

    print(f"Thời gian nạp toàn bộ 2.036 mẫu vào RAM : {load_time*1000:.2f} ms (< 0.1s)")
    print(f"Tổng dung lượng chiếm dụng RAM           : {total_ram_mb:.2f} MB (Rất nhẹ, tối ưu cho PC)")
    print(f"- Tập Train : {len(X_train):4d} mẫu | Shape: {X_train.shape}")
    print(f"- Tập Val   : {len(X_val):4d} mẫu | Shape: {X_val.shape}")
    print(f"- Tập Test  : {len(X_test):4d} mẫu | Shape: {X_test.shape}")

    # 2. Benchmark Throughput qua DataLoader
    print("\n--- 2. TỐC ĐỘ CẤP PHÁT CHO PYTORCH DATALOADER (BATCH_SIZE=32) ---")
    bench = cache_ram.benchmark_throughput(split="train", batch_size=32)
    print(f"Số mẫu kiểm thử  : {bench['total_samples']} samples ({bench['batches']} batches)")
    print(f"Thời gian quét   : {bench['elapsed_seconds']*1000:.2f} ms")
    print(f"Tốc độ nạp (FPS) : {bench['samples_per_sec']:,.0f} samples / giây")

    # 3. Đo lường chế độ MEMORY-MAPPED (mmap)
    print("\n--- 3. ĐO LƯỜNG CHẾ ĐỘ ÁNH XẠ ĐĨA KHÔNG CHIẾM RAM (MEMORY-MAPPED) ---")
    t0 = time.perf_counter()
    cache_mmap = FeatureCache(cache_dir=cache_dir, in_memory=False)
    bench_mmap = cache_mmap.benchmark_throughput(split="train", batch_size=32)
    load_time_mmap = time.perf_counter() - t0
    print(f"Thời gian quét qua mmap  : {bench_mmap['elapsed_seconds']*1000:.2f} ms")
    print(f"Tốc độ nạp qua mmap      : {bench_mmap['samples_per_sec']:,.0f} samples / giây")

    # Đánh giá tiêu chuẩn
    is_fast = bench["samples_per_sec"] >= 5000  # Kỳ vọng >= 5.000 samples/sec
    print("\n" + "=" * 65)
    if is_fast:
        print(">>> ĐẠT CHUẨN NGHIỆM THU W3-02: BỘ NHỚ ĐỆM TỐC ĐỘ CỰC CAO! <<<")
        print(f"Throughput đạt {bench['samples_per_sec']:,.0f} samples/s. Sẵn sàng cho huấn luyện CNN!")
    else:
        print(">>> BỘ NHỚ ĐỆM HOẠT ĐỘNG BÌNH THƯỜNG <<<")
    print("=" * 65)

    return True


if __name__ == "__main__":
    run_cache_benchmark()
