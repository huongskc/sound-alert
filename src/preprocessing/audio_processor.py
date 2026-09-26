"""
src/preprocessing/audio_processor.py

Module tiền xử lý tín hiệu âm thanh và trích xuất đặc trưng Log-Mel Spectrogram
chuẩn hóa cho hệ thống phát hiện sự kiện bất thường bằng Custom 2D-CNN.

Đặc tả kỹ thuật:
- Tần số lấy mẫu mục tiêu: 16.000 Hz
- Kênh: 1 (Mono)
- Thời lượng cố định: 2.0s (32.000 samples)
- FFT: n_fft=1024, win_length=1024, hop_length=512 (Hann window)
- Mel filterbank: 64 dải tần (50 Hz - 8000 Hz)
- Thang đo: Power Spectrogram chuyển đổi sang dB (top_db=80.0)
- Chuẩn hóa: Min-Max về khoảng [0, 1]
- Shape đầu ra chuẩn cho Custom 2D-CNN: [1, 64, 63] (Channel x Freq x Time)
"""

import os
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import yaml
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class AudioProcessor:
    """
    Engine xử lý tín hiệu âm thanh và trích xuất Log-Mel Spectrogram.
    Hỗ trợ cả xử lý ngoại tuyến (file WAV/FLAC) và luồng streaming (mảng numpy từ mic).
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        duration_seconds: float = 2.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: Optional[int] = 1024,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        top_db: float = 80.0,
        target_peak: float = 0.95,
        norm_mode: str = "min_max"
    ):
        """
        Khởi tạo AudioProcessor từ file YAML hoặc các tham số tùy chỉnh.
        """
        # Đọc cấu hình từ YAML nếu được cung cấp
        if config_path is not None:
            config_p = Path(config_path)
            if config_p.exists():
                with open(config_p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                audio_cfg = cfg.get("audio", {})
                mel_cfg = cfg.get("mel_spectrogram", {})
                
                sample_rate = audio_cfg.get("sample_rate", sample_rate)
                duration_seconds = audio_cfg.get("duration_seconds", duration_seconds)
                target_peak = audio_cfg.get("target_peak", target_peak)
                
                n_mels = mel_cfg.get("n_mels", n_mels)
                n_fft = mel_cfg.get("n_fft", n_fft)
                hop_length = mel_cfg.get("hop_length", hop_length)
                win_length = mel_cfg.get("win_length", win_length)
                f_min = mel_cfg.get("f_min", f_min)
                f_max = mel_cfg.get("f_max", f_max)
                top_db = mel_cfg.get("top_db", top_db)
                if mel_cfg.get("normalize_min_max", True):
                    norm_mode = "min_max"
                else:
                    norm_mode = "db_fixed"

        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.target_length = int(sample_rate * duration_seconds)
        self.target_peak = target_peak
        self.norm_mode = norm_mode

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.top_db = top_db

        # Khởi tạo MelSpectrogram transform từ torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0
        )

        # Chuyển đổi công suất sang Decibel (Log-Mel)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            top_db=self.top_db
        )

        # Cache cho các Resampler theo tần số nguồn
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Trả về resampler phù hợp với tần số lấy mẫu nguồn."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )
        return self._resamplers[orig_sr]

    def load_audio(self, audio_source: Union[str, Path, np.ndarray, torch.Tensor], sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Nạp tín hiệu âm thanh từ file hoặc mảng bộ nhớ.
        Trả về (waveform, sample_rate) với waveform dạng Tensor [channels, samples].
        """
        if isinstance(audio_source, (str, Path)):
            p = Path(audio_source)
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy file âm thanh: {p}")
            data, sr = sf.read(str(p), dtype="float32")
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [1, samples]
            else:
                waveform = waveform.t()  # [channels, samples]
            return waveform, sr

        elif isinstance(audio_source, np.ndarray):
            sr = sample_rate or self.sample_rate
            waveform = torch.from_numpy(audio_source).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                waveform = waveform.t()
            return waveform, sr

        elif isinstance(audio_source, torch.Tensor):
            sr = sample_rate or self.sample_rate
            waveform = audio_source.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            return waveform, sr

        else:
            raise TypeError(f"Dạng dữ liệu không được hỗ trợ: {type(audio_source)}")

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi tín hiệu nhiều kênh (stereo/multichannel) về mono 1 kênh."""
        if waveform.shape[0] > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample tín hiệu về tần số mục tiêu (mặc định 16.000 Hz)."""
        if orig_sr != self.sample_rate:
            resampler = self._get_resampler(orig_sr)
            return resampler(waveform)
        return waveform

    def normalize_peak(self, waveform: torch.Tensor) -> torch.Tensor:
        """Chuẩn hóa đỉnh biên độ tín hiệu về [-target_peak, target_peak] để tránh vỡ tiếng."""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-6:
            return waveform * (self.target_peak / max_val)
        return waveform

    def pad_or_crop(self, waveform: torch.Tensor, crop_mode: str = "center") -> torch.Tensor:
        """
        Cắt hoặc đệm tín hiệu về độ dài cố định target_length (32.000 samples).
        - Nếu ngắn hơn 2.0s: Đệm số 0 đối xứng 2 đầu (symmetric zero-padding).
        - Nếu dài hơn 2.0s: Cắt chính giữa (center-crop) để giữ sự kiện trung tâm.
        """
        num_samples = waveform.shape[-1]

        if num_samples == self.target_length:
            return waveform

        if num_samples < self.target_length:
            # Đệm đối xứng
            pad_left = (self.target_length - num_samples) // 2
            pad_right = self.target_length - num_samples - pad_left
            return F.pad(waveform, (pad_left, pad_right), mode="constant", value=0.0)

        # Cắt nếu dài hơn
        if crop_mode == "center":
            start = (num_samples - self.target_length) // 2
            return waveform[:, start:start + self.target_length]
        elif crop_mode == "start":
            return waveform[:, :self.target_length]
        elif crop_mode == "random":
            max_start = num_samples - self.target_length
            start = int(torch.randint(0, max_start + 1, (1,)).item())
            return waveform[:, start:start + self.target_length]
        else:
            return waveform[:, :self.target_length]

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất Log-Mel Spectrogram và chuẩn hóa dải động.
        Đầu vào: waveform [1, 32000]
        Đầu ra: Log-Mel Spectrogram tensor [1, 64, 63] chuẩn hóa trong khoảng [0, 1].
        """
        # 1. Tính toán Mel Spectrogram (Power)
        mel_spec = self.mel_transform(waveform)

        # 2. Chuyển sang Decibel
        mel_db = self.amplitude_to_db(mel_spec)

        # 3. Chuẩn hóa dải giá trị về [0, 1]
        if self.norm_mode == "min_max":
            min_val = mel_db.min()
            max_val = mel_db.max()
            diff = max_val - min_val
            if diff > 1e-6:
                mel_norm = (mel_db - min_val) / diff
            else:
                mel_norm = torch.zeros_like(mel_db)
        elif self.norm_mode == "db_fixed":
            # Ánh xạ tuyến tính [-top_db, 0] dB về [0, 1]
            mel_norm = torch.clamp((mel_db + self.top_db) / self.top_db, min=0.0, max=1.0)
        else:
            mel_norm = mel_db

        return mel_norm

    def process(
        self,
        audio_source: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        crop_mode: str = "center"
    ) -> torch.Tensor:
        """
        Toàn bộ chu trình tiền xử lý từ nguồn âm thanh đến Tensor ảnh phổ Log-Mel [1, 64, 63].
        
        Các bước:
        1. Nạp tín hiệu
        2. Chuyển về Mono 1 kênh
        3. Resample về 16.000 Hz
        4. Cắt/Đệm về đúng 2.0 giây (32.000 samples)
        5. Chuẩn hóa đỉnh biên độ [-0.95, 0.95]
        6. Trích xuất Log-Mel Spectrogram [1, 64, 63] trên khoảng [0, 1]
        """
        waveform, sr = self.load_audio(audio_source, sample_rate)
        waveform = self.to_mono(waveform)
        waveform = self.resample(waveform, sr)
        waveform = self.pad_or_crop(waveform, crop_mode=crop_mode)
        waveform = self.normalize_peak(waveform)
        mel_spec = self.extract_mel_spectrogram(waveform)
        return mel_spec

    def __call__(self, audio_source: Union[str, Path, np.ndarray, torch.Tensor], sample_rate: Optional[int] = None) -> torch.Tensor:
        return self.process(audio_source, sample_rate)
