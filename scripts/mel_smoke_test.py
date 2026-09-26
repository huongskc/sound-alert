"""
scripts/mel_smoke_test.py

PoC & Nghiệm thu trích xuất ảnh phổ Log-Mel Spectrogram (Mã W1-06):
1. Lấy mẫu đại diện từ cả 9 lớp sự kiện trong data/raw_categorized/.
2. Sử dụng AudioProcessor để trích xuất tensor Log-Mel Spectrogram chuẩn [1, 64, 63].
3. Kiểm tra shape, dải giá trị [0, 1], đo lường tốc độ xử lý trên CPU (ms/mẫu).
4. Vẽ biểu đồ so sánh phổ 3x3 trực quan và lưu vào reports/figures/mel_9_classes.png.
"""

import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_processor import AudioProcessor

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

CLASS_TITLES = {
    "1_glass_break": "1. Glass Break (Kính vỡ)",
    "2_gunshot_explosion": "2. Gunshot / Explosion (Súng / Nổ)",
    "3_door_forced": "3. Door Forced (Cậy cửa / Đập cửa)",
    "4_scream": "4. Scream (La hét hoảng loạn)",
    "5_cry_help": "5. Cry / Help (Tiếng khóc / Kêu cứu)",
    "6_fire_alarm_siren": "6. Fire Alarm / Siren (Còi báo động)",
    "7_loud_impact": "7. Loud Impact (Va đập mạnh)",
    "8_air_leak_hiss": "8. Air Leak / Hiss (Xì khí nén)",
    "9_background_normal": "9. Background Normal (Bình thường)"
}


def run_mel_smoke_test():
    print("=" * 65)
    print("      SMOKE TEST: TRÍCH XUẤT ĐẶC TRƯNG LOG-MEL SPECTROGRAM (W1-06)")
    print("=" * 65)

    config_path = PROJECT_ROOT / "configs" / "audio_config.yaml"
    processor = AudioProcessor(config_path=config_path)

    categorized_dir = PROJECT_ROOT / "data" / "raw_categorized"
    reports_fig_dir = PROJECT_ROOT / "reports" / "figures"
    reports_fig_dir.mkdir(parents=True, exist_ok=True)

    samples = {}
    latencies = []

    print("\n--- 1. KIỂM TRA ĐẦU RA TENSOR VÀ TỐC ĐỘ XỬ LÝ (CPU) ---")
    print(f"{'Lớp sự kiện':<25} | {'Shape Tensor':<16} | {'Min - Max':<14} | {'Độ trễ':<10}")
    print("-" * 72)

    for cname in CLASS_NAMES:
        cdir = categorized_dir / cname
        wav_files = list(cdir.glob("*.wav"))
        if not wav_files:
            print(f"[CẢNH BÁO] Không tìm thấy file trong {cname}")
            continue

        sample_file = wav_files[0]  # Chọn file đầu tiên làm đại diện

        t0 = time.perf_counter()
        mel_tensor = processor.process(sample_file)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        shape_str = str(list(mel_tensor.shape))
        min_v = mel_tensor.min().item()
        max_v = mel_tensor.max().item()
        range_str = f"[{min_v:.2f}, {max_v:.2f}]"

        assert mel_tensor.shape == (1, 64, 63), f"Lỗi shape: kỳ vọng (1, 64, 63) nhưng nhận {mel_tensor.shape}"
        assert 0.0 <= min_v and max_v <= 1.0001, f"Lỗi dải giá trị ngoài [0, 1]: min={min_v}, max={max_v}"

        print(f"{cname:<25} | {shape_str:<16} | {range_str:<14} | {latency_ms:6.2f} ms")
        samples[cname] = (sample_file.name, mel_tensor.squeeze(0).numpy())

    avg_latency = np.mean(latencies)
    print("-" * 72)
    print(f"-> Độ trễ xử lý trung bình trên CPU: {avg_latency:.2f} ms / file (Mục tiêu: < 30 ms)")
    if avg_latency < 30.0:
        print("[ĐẠT YÊU CẦU] Tốc độ xử lý đặc trưng hoàn toàn đáp ứng thời gian thực (Real-time).")

    # Vẽ lưới 3x3 trực quan hóa
    print("\n--- 2. TẠO BIỂU ĐỒ TRỰC QUAN HÓA PHỔ 9 LỚP ---")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Log-Mel Spectrogram Comparison (9 Audio Event Classes)\nTarget Shape: [64 Mel Bins x 63 Time Steps] (Normalized [0, 1])", fontsize=15, fontweight="bold")

    time_extent = [0.0, 2.0, 50, 8000]  # Time: 0-2s, Freq: 50-8000Hz

    for idx, cname in enumerate(CLASS_NAMES):
        ax = axes[idx // 3, idx % 3]
        fname, mel_data = samples[cname]

        im = ax.imshow(
            mel_data,
            origin="lower",
            aspect="auto",
            extent=time_extent,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0
        )
        ax.set_title(CLASS_TITLES[cname], fontsize=11, fontweight="bold")
        ax.set_xlabel("Thời gian (giây)", fontsize=9)
        ax.set_ylabel("Tần số (Hz)", fontsize=9)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Cường độ phổ chuẩn hóa [0, 1]", fontsize=11)

    out_fig = reports_fig_dir / "mel_9_classes.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[HOÀN TẤT] Đã lưu biểu đồ trực quan hóa tại: {out_fig}")

    print("\n" + "=" * 65)
    print(">>> TỔNG KẾT: SMOKE TEST LOG-MEL SPECTROGRAM THÀNH CÔNG 100%! <<<")
    print("=" * 65)


if __name__ == "__main__":
    run_mel_smoke_test()
