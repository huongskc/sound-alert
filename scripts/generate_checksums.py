"""
scripts/generate_checksums.py

Tạo và kiểm tra mã băm MD5 đóng băng tập dữ liệu dataset_v1 (Mã W2-10):
1. Tính mã băm MD5 cho từng tệp âm thanh trong data/raw_categorized/ (2.036 tệp).
2. Lưu kết quả vào data/metadata/dataset_v1_checksums.md5.
3. Hỗ trợ cờ --verify để đối soát lại toàn vẹn bất kỳ lúc nào.
"""

import sys
import hashlib
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATEGORIZED_DIR = PROJECT_ROOT / "data" / "raw_categorized"
CHECKSUM_FILE = PROJECT_ROOT / "data" / "metadata" / "dataset_v1_checksums.md5"


def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Tính toán mã băm MD5 cho một tệp."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_checksums():
    print("=" * 65)
    print("      TẠO MÃ BĂM MD5 ĐÓNG BĂNG TẬP DỮ LIỆU DATASET_V1 (W2-10)")
    print("=" * 65)

    if not CATEGORIZED_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục {CATEGORIZED_DIR}")
        return False

    all_files = sorted(list(CATEGORIZED_DIR.rglob("*.wav")))
    total_files = len(all_files)
    print(f"[INFO] Bắt đầu tính mã băm MD5 cho {total_files} tệp âm thanh...")

    checksum_lines = []
    total_bytes = 0

    for idx, fpath in enumerate(all_files, start=1):
        md5_hash = compute_file_md5(fpath)
        rel_path = fpath.relative_to(PROJECT_ROOT).as_posix()
        checksum_lines.append(f"{md5_hash}  {rel_path}")
        total_bytes += fpath.stat().st_size

        if idx % 500 == 0 or idx == total_files:
            pct = (idx / total_files) * 100
            sys.stdout.write(f"\r[TIẾN ĐỘ] Đã tính: {idx}/{total_files} ({pct:5.1f}%)")
            sys.stdout.flush()

    print()
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print(f"[HOÀN TẤT] Đã ghi mã băm của {total_files} tệp vào: {CHECKSUM_FILE}")
    print(f"[THỐNG KÊ] Tổng dung lượng dữ liệu: {total_bytes / (1024*1024):.2f} MB")
    print("\n" + "=" * 65)
    print(">>> TẬP DỮ LIỆU DATASET_V1 CHÍNH THỨC ĐƯỢC ĐÓNG BĂNG (MỐC M2)! <<<")
    print("=" * 65)
    return True


def verify_checksums():
    print("=" * 65)
    print("      ĐỐI SOÁT MÃ BĂM MD5 TẬP DỮ LIỆU DATASET_V1")
    print("=" * 65)

    if not CHECKSUM_FILE.exists():
        print(f"[ERROR] Không tìm thấy tệp checksum tại {CHECKSUM_FILE}")
        return False

    with open(CHECKSUM_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Đối soát {len(lines)} tệp từ file checksum...")
    mismatch_cnt = 0
    missing_cnt = 0

    for idx, line in enumerate(lines, start=1):
        expected_hash, rel_path = line.split("  ", 1)
        full_path = PROJECT_ROOT / rel_path

        if not full_path.exists():
            missing_cnt += 1
            continue

        actual_hash = compute_file_md5(full_path)
        if actual_hash != expected_hash:
            mismatch_cnt += 1

        if idx % 500 == 0 or idx == len(lines):
            pct = (idx / len(lines)) * 100
            sys.stdout.write(f"\r[ĐỐI SOÁT] {idx}/{len(lines)} ({pct:5.1f}%) - Lỗi: {mismatch_cnt + missing_cnt}")
            sys.stdout.flush()

    print()
    is_ok = (mismatch_cnt == 0 and missing_cnt == 0)
    print(f"File thiếu       : {missing_cnt}")
    print(f"File sai mã băm  : {mismatch_cnt}")
    if is_ok:
        print("[KẾT QUẢ] 100% tệp âm thanh trùng khớp chính xác mã băm gốc!")
    else:
        print("[LỖI] Dữ liệu có sự thay đổi hoặc thiếu tệp!")
    return is_ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        success = verify_checksums()
    else:
        success = generate_checksums()
    if not success:
        sys.exit(1)
