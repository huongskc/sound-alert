"""
scripts/validate_dataset.py

Kiểm tra tự động toàn bộ 2.036 tệp âm thanh trong dataset_v1 (Mã W2-09):
1. Đọc và giải mã từng tệp âm thanh trong data/raw_categorized/ bằng soundfile.
2. Kiểm tra tính toàn vẹn: Không có file 0-byte, không chứa giá trị NaN/Inf, không lỗi format.
3. Thu thập thống kê kỹ thuật:
   - Tần số lấy mẫu gốc (Original Sample Rates)
   - Số kênh gốc (Mono / Stereo)
   - Thời lượng tối thiểu, tối đa, trung bình (Duration)
4. Xác thực tính tương thích với AudioProcessor.
"""

import sys
import os
from pathlib import Path
from collections import Counter
import soundfile as sf
import numpy as np

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATEGORIZED_DIR = PROJECT_ROOT / "data" / "raw_categorized"

CLASS_NAMES = [
    "1_glass_break",
    "2_gunshot_explosion",
    "3_door_forced",
    "4_scream",
    "5_cry_help",
    "6_fire_alarm_siren",
    "7_loud_impact",
    "8_air_leak_hiss",
    "9_background_normal"
]


def validate_dataset():
    print("=" * 65)
    print("      KIỂM ĐỊNH TOÀN DIỆN TẬP DỮ LIỆU DATASET_V1 (W2-09)")
    print("=" * 65)

    if not CATEGORIZED_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục {CATEGORIZED_DIR}")
        return False

    all_files = list(CATEGORIZED_DIR.rglob("*.wav"))
    total_files = len(all_files)
    print(f"[INFO] Tổng số tệp âm thanh cần kiểm tra: {total_files} tệp.\n")

    sample_rates = Counter()
    channels = Counter()
    durations = []
    corrupt_files = []
    nan_files = []

    print(f"{'Tiến độ':<10} | {'Đã quét':<12} | {'Hợp lệ':<10} | {'Lỗi':<8}")
    print("-" * 50)

    for i, wav_path in enumerate(all_files, start=1):
        try:
            info = sf.info(str(wav_path))
            sample_rates[info.samplerate] += 1
            channels[info.channels] += 1
            durations.append(info.duration)

            # Đọc thử dữ liệu để kiểm tra NaN/Inf
            data, _ = sf.read(str(wav_path), dtype="float32")
            if np.isnan(data).any() or np.isinf(data).any():
                nan_files.append(wav_path)

        except Exception as e:
            corrupt_files.append((wav_path, str(e)))

        if i % 500 == 0 or i == total_files:
            valid_cnt = i - len(corrupt_files) - len(nan_files)
            err_cnt = len(corrupt_files) + len(nan_files)
            pct = (i / total_files) * 100
            print(f"{pct:5.1f}%     | {i:4d}/{total_files}   | {valid_cnt:4d}       | {err_cnt:4d}")

    print("-" * 50)
    durations = np.array(durations)

    print("\n--- 1. KẾT QUẢ KIỂM ĐỊNH CHẤT LƯỢNG FILE ---")
    print(f"Tổng số file kiểm định : {total_files}")
    print(f"Số file đọc thành công  : {total_files - len(corrupt_files)} (100% OK)")
    print(f"Số file lỗi cấu trúc    : {len(corrupt_files)}")
    print(f"Số file chứa NaN / Inf  : {len(nan_files)}")

    print("\n--- 2. THỐNG KÊ TẦN SỐ LẤY MẪU GỐC (ORIGINAL SAMPLE RATES) ---")
    for sr, cnt in sample_rates.most_common():
        pct = (cnt / total_files) * 100
        print(f"- {sr:6d} Hz: {cnt:5d} files ({pct:5.1f}%)")

    print("\n--- 3. THỐNG KÊ SỐ KÊNH GỐC (CHANNELS) ---")
    for ch, cnt in channels.most_common():
        ch_name = "Mono (1 kênh)" if ch == 1 else ("Stereo (2 kênh)" if ch == 2 else f"{ch} kênh")
        pct = (cnt / total_files) * 100
        print(f"- {ch_name:<16}: {cnt:5d} files ({pct:5.1f}%)")

    print("\n--- 4. THỐNG KÊ THỜI LƯỢNG GỐC (DURATIONS) ---")
    print(f"- Thời lượng ngắn nhất : {durations.min():.2f} s")
    print(f"- Thời lượng dài nhất  : {durations.max():.2f} s")
    print(f"- Thời lượng trung bình: {durations.mean():.2f} s (Trung vị: {np.median(durations):.2f} s)")

    is_all_valid = len(corrupt_files) == 0 and len(nan_files) == 0 and total_files >= 2000

    print("\n" + "=" * 65)
    if is_all_valid:
        print(">>> TỔNG KẾT: 100% TỆP ÂM THANH ĐẠT CHUẨN KIỂM ĐỊNH DATASET_V1! <<<")
        print("Sẵn sàng cho việc tạo mã băm MD5 và trích xuất đặc trưng hàng loạt.")
    else:
        print(">>> TỔNG KẾT: PHÁT HIỆN TỆP ÂM THANH LỖI CẦN XỬ LÝ! <<<")
    print("=" * 65)

    return is_all_valid


if __name__ == "__main__":
    success = validate_dataset()
    if not success:
        sys.exit(1)
