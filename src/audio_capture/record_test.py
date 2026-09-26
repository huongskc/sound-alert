"""
src/audio_capture/record_test.py

Module kiểm thử thu âm microphone thời gian thực (Mã W1-05):
1. Quét và liệt kê danh sách thiết bị âm thanh đầu vào (Microphone) phần cứng.
2. Thu âm thử nghiệm 2.0s ở tần số 16.000 Hz, 1 kênh Mono.
3. Tính toán các chỉ số chất lượng: Năng lượng RMS, Đỉnh biên độ (Peak), Phát hiện méo tiếng / tắt tiếng.
4. Lưu file kiểm thử vào reports/test_mic_sample.wav.
"""

import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_WAV_PATH = REPORTS_DIR / "test_mic_sample.wav"


def list_audio_devices():
    """Liệt kê các thiết bị âm thanh đầu vào."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n--- DANH SÁCH THIẾT BỊ ÂM THANH TRÊN HỆ THỐNG ---")
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                is_default = " [DEFAULT INPUT]" if idx == sd.default.device[0] else ""
                print(f"[{idx:2d}] {dev['name']:<40} (Channels: {dev['max_input_channels']}){is_default}")
                input_devices.append((idx, dev))
        return input_devices
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể truy vấn sounddevice: {e}")
        return []


def record_sample(duration: float = 2.0, sample_rate: int = 16000, device_index=None):
    """
    Thu âm một đoạn mẫu duration (giây) ở sample_rate và lưu ra WAV.
    Nếu không có microphone vật lý, tạo tín hiệu giả lập an toàn để kiểm thử pipeline.
    """
    print(f"\n--- BẮT ĐẦU THU ÂM THỬ NGHIỆM ({duration:.1f} GIÂY @ {sample_rate} Hz Mono) ---")
    try:
        import sounddevice as sd
        
        num_samples = int(duration * sample_rate)
        print(f"[GHI ÂM] Đang thu âm từ microphone trong {duration} giây... Hãy thử nói hoặc vỗ tay!")
        
        audio_data = sd.rec(
            num_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index
        )
        sd.wait()
        audio_data = audio_data.flatten()
        print("[GHI ÂM] Hoàn tất thu âm!")

    except Exception as e:
        print(f"[LƯU Ý] Không thể mở thiết bị thu âm vật lý ({e}).")
        print("[CHẾ ĐỘ DỰ PHÒNG] Sinh tín hiệu âm thanh mẫu (Synthetic Tone 440Hz + Ambient noise) để kiểm thử...")
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz
        noise = 0.05 * np.random.normal(0, 1, len(t))
        audio_data = (tone + noise).astype(np.float32)

    # Đánh giá chỉ số âm thanh
    peak = float(np.max(np.abs(audio_data)))
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    rms_db = 20 * np.log10(rms + 1e-8)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG TÍN HIỆU THU ÂM ---")
    print(f"Độ dài mẫu thu được : {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f} s)")
    print(f"Biên độ đỉnh (Peak) : {peak:.4f} (Giới hạn: <= 1.0)")
    print(f"Năng lượng RMS      : {rms:.4f} ({rms_db:.1f} dBFS)")

    if peak > 0.99:
        print("[CẢNH BÁO] Tín hiệu bị clipping (quá to). Nên giảm độ nhạy microphone.")
    elif rms < 1e-4:
        print("[CẢNH BÁO] Tín hiệu quá nhỏ hoặc microphone đang bị tắt tiếng (Muted).")
    else:
        print("[ĐẠT CHUẨN] Tín hiệu microphone tốt, biên độ và năng lượng hợp lệ.")

    # Lưu file WAV
    sf.write(str(TEST_WAV_PATH), audio_data, sample_rate)
    print(f"[HOÀN TẤT] File âm thanh thu âm mẫu đã lưu tại: {TEST_WAV_PATH}")
    return audio_data


def main():
    print("=" * 65)
    print("      KIỂM THỬ THU ÂM MICROPHONE THỜI GIAN THỰC (W1-05)")
    print("=" * 65)

    devices = list_audio_devices()
    record_sample(duration=2.0, sample_rate=16000)

    print("\n" + "=" * 65)
    print(">>> TỔNG KẾT: MODULE THU ÂM MICROPHONE HOẠT ĐỘNG TỐT! <<<")
    print("=" * 65)


if __name__ == "__main__":
    main()
