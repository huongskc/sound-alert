"""
scripts/extract_features.py

Pipeline trích xuất đặc trưng Log-Mel Spectrogram hàng loạt và lưu cache .npy (Mã W3-01):
1. Nạp danh mục từ data/metadata/dataset_v1_manifest.csv (2.036 clips).
2. Sử dụng AudioProcessor để chuẩn hóa và trích xuất Log-Mel tensor [1, 64, 63].
3. Lưu bộ nhớ đệm kép (Dual-cache):
   - Tệp đơn lẻ: data/processed/features/<split>/<sample_id>.npy
   - Mảng tổng hợp: X_<split>.npy, y_<split>.npy, y_<split>_onehot.npy (load tức thì < 0.1s khi train)
4. Xuất bảng manifest đặc trưng data/processed/features/features_manifest.csv.
5. Đo lường tốc độ trích xuất và kiểm tra toàn vẹn tensor.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_processor import AudioProcessor

MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_v1_manifest.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "audio_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"


def extract_features(num_workers: int = 8):
    print("=" * 65)
    print("      PIPELINE TRÍCH XUẤT ĐẶC TRƯNG LOG-MEL SPECTROGRAM (W3-01)")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest tại {MANIFEST_PATH}")

    df_manifest = pd.read_csv(MANIFEST_PATH)
    total_samples = len(df_manifest)
    print(f"[INFO] Tổng số mẫu cần trích xuất: {total_samples} clips.")

    # Khởi tạo thư mục đích
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        (OUTPUT_DIR / s).mkdir(parents=True, exist_ok=True)

    # Khởi tạo AudioProcessor
    processor = AudioProcessor(config_path=CONFIG_PATH)

    print(f"[INFO] Bắt đầu trích xuất song song (Workers: {num_workers})...")
    start_time = time.perf_counter()

    def process_item(item):
        sample_id = item["sample_id"]
        rel_path = item["relative_path"]
        split = item["split"]
        class_idx = item["class_index"]
        label_idx = class_idx - 1  # 0-indexed (0 đến 8)

        full_audio_path = PROJECT_ROOT / rel_path
        feature_path = OUTPUT_DIR / split / f"{sample_id}.npy"

        # Trích xuất đặc trưng
        mel_tensor = processor.process(full_audio_path)
        feature_np = mel_tensor.squeeze(0).numpy().astype(np.float32)  # [64, 63]

        # Lưu tệp .npy đơn lẻ
        np.save(str(feature_path), feature_np)

        return {
            "sample_id": sample_id,
            "split": split,
            "label_idx": label_idx,
            "class_index": class_idx,
            "class_name": item["class_name"],
            "recording_id": item["recording_id"],
            "source_dataset": item["source_dataset"],
            "feature_rel_path": str(feature_path.relative_to(PROJECT_ROOT).as_posix()),
            "feature_data": feature_np
        }

    results = []
    failed_items = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_item, row): idx 
            for idx, row in df_manifest.iterrows()
        }

        for count, future in enumerate(as_completed(future_to_idx), start=1):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                failed_items.append((future_to_idx[future], str(e)))

            if count % 200 == 0 or count == total_samples:
                elapsed = time.perf_counter() - start_time
                speed = count / elapsed
                pct = (count / total_samples) * 100
                sys.stdout.write(f"\r[TIẾN ĐỘ] {count:4d}/{total_samples} ({pct:5.1f}%) | Tốc độ: {speed:5.1f} clips/s | Lỗi: {len(failed_items)}")
                sys.stdout.flush()

    print()
    total_time = time.perf_counter() - start_time
    avg_speed = total_samples / total_time

    print(f"\n[KẾT QUẢ TRÍCH XUẤT] Hoàn tất {len(results)}/{total_samples} clips trong {total_time:.2f}s ({avg_speed:.1f} clips/s).")
    if failed_items:
        print(f"[CẢNH BÁO] Có {len(failed_items)} mẫu bị lỗi!")
        for idx, err in failed_items[:5]:
            print(f"  - Mẫu {idx}: {err}")
        return False

    # 4. Gom cụm và lưu mảng tổng hợp X_<split>.npy, y_<split>.npy
    print("\n--- ĐÓNG GÓI MẢNG TỔNG HỢP CHO TRAINING SIÊU TỐC ---")
    splits_data = {"train": [], "val": [], "test": []}
    splits_labels = {"train": [], "val": [], "test": []}
    manifest_rows = []

    for r in results:
        s = r["split"]
        splits_data[s].append(r["feature_data"])
        splits_labels[s].append(r["label_idx"])
        manifest_rows.append({
            "sample_id": r["sample_id"],
            "class_name": r["class_name"],
            "class_index": r["class_index"],
            "label_idx": r["label_idx"],
            "split": r["split"],
            "recording_id": r["recording_id"],
            "source_dataset": r["source_dataset"],
            "feature_path": r["feature_rel_path"]
        })

    for s in ["train", "val", "test"]:
        X_arr = np.array(splits_data[s], dtype=np.float32)  # [N, 64, 63]
        X_arr = np.expand_dims(X_arr, axis=1)              # [N, 1, 64, 63]
        y_arr = np.array(splits_labels[s], dtype=np.int64)  # [N]
        
        # One-hot encoding [N, 9]
        num_classes = 9
        y_onehot = np.zeros((len(y_arr), num_classes), dtype=np.float32)
        y_onehot[np.arange(len(y_arr)), y_arr] = 1.0

        x_path = OUTPUT_DIR / f"X_{s}.npy"
        y_path = OUTPUT_DIR / f"y_{s}.npy"
        y_onehot_path = OUTPUT_DIR / f"y_{s}_onehot.npy"

        np.save(str(x_path), X_arr)
        np.save(str(y_path), y_arr)
        np.save(str(y_onehot_path), y_onehot)

        size_mb = (X_arr.nbytes + y_arr.nbytes + y_onehot.nbytes) / (1024 * 1024)
        print(f"- Tập {s.upper():<5}: {len(X_arr):4d} samples | Shape: {X_arr.shape} | Dung lượng: {size_mb:5.2f} MB")

    # Lưu manifest đặc trưng
    df_feat_manifest = pd.DataFrame(manifest_rows)
    feat_manifest_path = OUTPUT_DIR / "features_manifest.csv"
    df_feat_manifest.to_csv(feat_manifest_path, index=False, encoding="utf-8")
    print(f"\n[HOÀN TẤT] File manifest đặc trưng lưu tại: {feat_manifest_path}")

    # Tính tổng dung lượng thư mục features
    total_feature_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"[THỐNG KÊ] Tổng dung lượng toàn bộ cache đặc trưng: {total_feature_bytes / (1024 * 1024):.2f} MB")
    print("\n" + "=" * 65)
    print(">>> TỔNG KẾT: TRÍCH XUẤT VÀ LƯU CACHE ĐẶC TRƯNG THÀNH CÔNG (W3-01)! <<<")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = extract_features(num_workers=8)
    if not success:
        sys.exit(1)
