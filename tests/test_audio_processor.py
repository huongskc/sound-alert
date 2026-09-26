"""
tests/test_audio_processor.py

Bộ kiểm thử đơn vị tự động (Unit Tests) cho AudioProcessor:
- Kiểm tra nạp cấu hình YAML
- Kiểm tra chuyển đổi Mono (1D, 2D stereo, multichannel)
- Kiểm tra cơ chế Resample từ các tần số khác nhau về 16kHz
- Kiểm tra cắt / đệm (Pad / Crop) về 32.000 samples (2.0s)
- Kiểm tra chuẩn hóa biên độ đỉnh (Peak Normalization)
- Kiểm tra shape đầu ra của Log-Mel Spectrogram [1, 64, 63] và dải giá trị [0, 1]
- Kiểm tra tính xác định (Deterministic output)
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_processor import AudioProcessor


@pytest.fixture
def processor():
    config_path = PROJECT_ROOT / "configs" / "audio_config.yaml"
    return AudioProcessor(config_path=config_path)


def test_initialization(processor):
    """Kiểm tra khởi tạo và các tham số cấu hình."""
    assert processor.sample_rate == 16000
    assert processor.duration_seconds == 2.0
    assert processor.target_length == 32000
    assert processor.n_mels == 64
    assert processor.n_fft == 1024
    assert processor.hop_length == 512
    assert processor.target_peak == 0.95


def test_to_mono(processor):
    """Kiểm tra chuyển đổi tín hiệu stereo và đa kênh về mono 1 kênh."""
    # 1. Tín hiệu mono 1 kênh [1, 1000]
    mono_in = torch.randn(1, 1000)
    mono_out = processor.to_mono(mono_in)
    assert mono_out.shape == (1, 1000)
    assert torch.allclose(mono_in, mono_out)

    # 2. Tín hiệu stereo 2 kênh [2, 1000]
    stereo_in = torch.randn(2, 1000)
    stereo_out = processor.to_mono(stereo_in)
    assert stereo_out.shape == (1, 1000)
    expected = torch.mean(stereo_in, dim=0, keepdim=True)
    assert torch.allclose(stereo_out, expected)


def test_resample(processor):
    """Kiểm tra resample từ các tần số lấy mẫu phổ biến về 16.000 Hz."""
    orig_rates = [44100, 48000, 22050]
    duration = 1.0  # 1 giây

    for sr in orig_rates:
        num_samples = int(sr * duration)
        waveform = torch.randn(1, num_samples)
        resampled = processor.resample(waveform, orig_sr=sr)
        # Kỳ vọng xấp xỉ 16.000 samples
        assert abs(resampled.shape[-1] - 16000) <= 2, f"Lỗi resample từ {sr} Hz: nhận {resampled.shape}"


def test_pad_or_crop_short_signal(processor):
    """Kiểm tra tín hiệu ngắn hơn 2.0s được đệm số 0 về đúng 32.000 samples."""
    short_waveform = torch.ones(1, 16000)  # 1.0s
    padded = processor.pad_or_crop(short_waveform)
    assert padded.shape == (1, 32000)
    # Kiểm tra phần đệm bên trái và phải là 0
    assert padded[0, 0].item() == 0.0
    assert padded[0, -1].item() == 0.0
    # Phần giữa phải giữ nguyên giá trị 1.0
    assert padded[0, 16000].item() == 1.0


def test_pad_or_crop_long_signal(processor):
    """Kiểm tra tín hiệu dài hơn 2.0s được center-crop về đúng 32.000 samples."""
    long_waveform = torch.randn(1, 64000)  # 4.0s
    cropped = processor.pad_or_crop(long_waveform, crop_mode="center")
    assert cropped.shape == (1, 32000)
    # Kiểm tra chính xác đoạn center đã được cắt
    expected = long_waveform[:, 16000:48000]
    assert torch.allclose(cropped, expected)


def test_pad_or_crop_exact_signal(processor):
    """Kiểm tra tín hiệu đúng 32.000 samples không bị biến đổi."""
    exact_waveform = torch.randn(1, 32000)
    result = processor.pad_or_crop(exact_waveform)
    assert result.shape == (1, 32000)
    assert torch.allclose(result, exact_waveform)


def test_normalize_peak(processor):
    """Kiểm tra chuẩn hóa biên độ đỉnh về 0.95."""
    waveform = torch.tensor([[0.0, 0.5, -2.0, 1.0]])
    normed = processor.normalize_peak(waveform)
    assert torch.isclose(torch.max(torch.abs(normed)), torch.tensor(0.95))

    # Kiểm tra trường hợp tín hiệu toàn 0 (im lặng)
    silent = torch.zeros(1, 1000)
    normed_silent = processor.normalize_peak(silent)
    assert torch.allclose(normed_silent, silent)


def test_extract_mel_spectrogram_shape_and_range(processor):
    """Kiểm tra shape [1, 64, 63] và dải giá trị [0, 1] của Log-Mel Spectrogram."""
    dummy_audio = torch.randn(1, 32000)
    mel = processor.extract_mel_spectrogram(dummy_audio)

    assert mel.shape == (1, 64, 63), f"Sai shape: kỳ vọng (1, 64, 63), nhận {mel.shape}"
    assert mel.min().item() >= 0.0
    assert mel.max().item() <= 1.0001


def test_deterministic_output(processor):
    """Kiểm tra tính xác định: cùng một đầu vào phải cho ra cùng một đầu ra 100%."""
    dummy_audio = torch.randn(1, 32000)
    out1 = processor.extract_mel_spectrogram(dummy_audio)
    out2 = processor.extract_mel_spectrogram(dummy_audio)
    assert torch.allclose(out1, out2)


def test_process_real_file(processor):
    """Kiểm tra xử lý trực tiếp file thực tế trong dataset_v1."""
    wav_path = PROJECT_ROOT / "data" / "raw_categorized" / "1_glass_break"
    files = list(wav_path.glob("*.wav"))
    if not files:
        pytest.skip("Chưa có file âm thanh để test")

    sample_file = files[0]
    mel_tensor = processor.process(sample_file)

    assert mel_tensor.shape == (1, 64, 63)
    assert 0.0 <= mel_tensor.min().item() <= 1.0
    assert 0.0 <= mel_tensor.max().item() <= 1.0
