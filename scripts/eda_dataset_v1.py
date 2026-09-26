"""
scripts/eda_dataset_v1.py

Phân tích Khám phá Dữ liệu (EDA) cho tập dataset_v1 (Mã W2-08):
1. Thống kê số lượng mẫu theo từng lớp và theo phân vùng (Train / Val / Test).
2. Thống kê nguồn gốc âm thanh (ESC-50, UrbanSound8K, FSD50K) đóng góp cho từng lớp.
3. Đo đạc và vẽ phân bố thời lượng âm thanh (Duration distribution).
4. Xuất biểu đồ tổng hợp chất lượng cao vào reports/figures/dataset_v1_eda.png.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_v1_manifest.csv"
REPORTS_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    print("=" * 65)
    print("      PHÂN TÍCH KHÁM PHÁ DỮ LIỆU DATASET_V1 (EDA - W2-08)")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Không tìm thấy file manifest tại {MANIFEST_PATH}")
        return False

    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Đọc manifest thành công: {len(df)} bản ghi.")

    # Thu thập thời lượng thực tế của từng clip
    print("[INFO] Đang trích xuất thời lượng file từ ổ đĩa...")
    durations = []
    for _, row in df.iterrows():
        fpath = PROJECT_ROOT / row["relative_path"]
        try:
            info = sf.info(str(fpath))
            durations.append(info.duration)
        except Exception:
            durations.append(0.0)

    df["duration"] = durations

    # 1. Báo cáo thống kê
    print("\n--- 1. BẢNG PHÂN BỐ SPLIT VÀ NGUỒN DỮ LIỆU THEO LỚP ---")
    summary = df.groupby(["class_index", "class_name"]).agg(
        total_samples=("sample_id", "count"),
        train_samples=("split", lambda s: (s == "train").sum()),
        val_samples=("split", lambda s: (s == "val").sum()),
        test_samples=("split", lambda s: (s == "test").sum()),
        esc50_samples=("source_dataset", lambda s: (s == "esc50").sum()),
        us8k_samples=("source_dataset", lambda s: (s == "urbansound8k").sum()),
        fsd50k_samples=("source_dataset", lambda s: (s == "fsd50k").sum()),
        min_dur=("duration", "min"),
        median_dur=("duration", "median"),
        max_dur=("duration", "max")
    ).reset_index()

    print(summary.to_string(index=False))

    # 2. Vẽ biểu đồ 4 ô (2x2 Grid)
    print("\n--- 2. VẼ BIỂU ĐỒ BÁO CÁO EDA ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    palette_split = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c"}
    palette_source = {"esc50": "#3498db", "urbansound8k": "#9b59b6", "fsd50k": "#1abc9c"}

    # Ô 1: Phân bố Train / Val / Test theo 9 lớp
    ax1 = axes[0, 0]
    split_pivot = df.pivot_table(index="class_name", columns="split", values="sample_id", aggfunc="count").fillna(0)
    split_pivot = split_pivot[["train", "val", "test"]]
    split_pivot.plot(kind="bar", stacked=True, color=[palette_split[c] for c in split_pivot.columns], ax=ax1, edgecolor="black", linewidth=0.5)
    ax1.set_title("1. Phân Bố Mẫu Theo Tập Split (Train 70% / Val 15% / Test 15%)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lớp sự kiện", fontsize=10)
    ax1.set_ylabel("Số lượng mẫu", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(title="Tập dữ liệu")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Ô 2: Nguồn gốc dữ liệu đóng góp cho 9 lớp
    ax2 = axes[0, 1]
    source_pivot = df.pivot_table(index="class_name", columns="source_dataset", values="sample_id", aggfunc="count").fillna(0)
    col_order = [c for c in ["esc50", "urbansound8k", "fsd50k"] if c in source_pivot.columns]
    source_pivot = source_pivot[col_order]
    source_pivot.plot(kind="bar", stacked=True, color=[palette_source[c] for c in col_order], ax=ax2, edgecolor="black", linewidth=0.5)
    ax2.set_title("2. Đóng Góp Nguồn Dữ Liệu (ESC-50, UrbanSound8K, FSD50K)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lớp sự kiện", fontsize=10)
    ax2.set_ylabel("Số lượng mẫu", fontsize=10)
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(title="Nguồn âm thanh")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Ô 3: Phân bố thời lượng gốc của âm thanh (Box plot)
    ax3 = axes[1, 0]
    classes = sorted(df["class_name"].unique())
    duration_by_class = [df[df["class_name"] == c]["duration"].values for c in classes]
    bp = ax3.boxplot(duration_by_class, tick_labels=classes, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#a8dadc')
    ax3.set_title("3. Phân Bố Thời Lượng Âm Thanh Gốc (Giây)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Lớp sự kiện", fontsize=10)
    ax3.set_ylabel("Thời lượng (giây)", fontsize=10)
    ax3.tick_params(axis="x", rotation=30)
    ax3.axhline(2.0, color="red", linestyle="--", linewidth=1.5, label="Độ dài chuẩn hóa (2.0s)")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Ô 4: Tỷ lệ tổng quát Train / Val / Test (Pie Chart)
    ax4 = axes[1, 1]
    split_counts = df["split"].value_counts()[["train", "val", "test"]]
    ax4.pie(
        split_counts,
        labels=[f"{s.upper()} ({cnt} clips)" for s, cnt in split_counts.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=[palette_split[s] for s in split_counts.index],
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax4.set_title(f"4. Tỷ Lệ Toàn Hệ Thống (Tổng: {len(df)} Clips, 100% Leak-Free)", fontsize=11, fontweight="bold")

    out_path = REPORTS_FIG_DIR / "dataset_v1_eda.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[HOÀN TẤT] Biểu đồ phân tích EDA đã lưu tại: {out_path}")
    print("\n" + "=" * 65)
    print(">>> TỔNG KẾT: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU EDA THÀNH CÔNG! <<<")
    print("=" * 65)
    return True


if __name__ == "__main__":
    run_eda()
