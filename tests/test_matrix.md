# DANH MỤC CA KIỂM THỬ VÀ MẪU BIÊN BẢN NGHIỆM THU (TEST MATRIX)

**Dự án:** Hệ thống giám sát và cảnh báo sự kiện bất thường dựa trên âm thanh sử dụng học sâu  
**Mã tài liệu:** `QA-TM-2026`  
**Phiên bản:** `1.0.0`  
**Người lập:** TV3 (UI, QA & Documentation Lead)  
**Phê duyệt:** Cả nhóm  

---

## 1. Tổng Quan Ma Trận Kiểm Thử

Ma trận kiểm thử bao quát toàn bộ 4 cấp độ kiểm thử của hệ thống:
1. **Unit Testing (UT):** Kiểm thử từng module độc lập (Xử lý âm thanh, trích xuất đặc trưng, hàm tính toán).
2. **Integration Testing (IT):** Kiểm thử liên kết giữa Microphone, Buffer, CNN Model, Alert Engine và SQLite.
3. **System Testing (ST):** Đánh giá hiệu năng thời gian thực trên CPU, độ trễ end-to-end, tiêu hao bộ nhớ.
4. **Model Performance Acceptance (MPA):** Tiêu chí định lượng của mô hình (Macro-F1, Recall các lớp khẩn cấp, False Alarm Rate).

---

## 2. Danh Mục Chi Tiết Các Ca Kiểm Thử

### 2.1. Cấp độ 1: Tiền Xử Lý Tín Hiệu Âm Thanh (Audio Preprocessing - UT-01 -> UT-08)

| Mã test | Tên ca kiểm thử | Mô tả & Đầu vào | Kết quả kỳ vọng | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **UT-01** | Khởi tạo cấu hình AudioProcessor | Nạp từ `configs/audio_config.yaml` | Đúng tham số: SR=16kHz, Win=1024, Hop=512, Mel=64, Duration=2.0s | **PASS** |
| **UT-02** | Chuyển đổi kênh Mono | Đầu vào stereo 2 kênh `[2, 32000]` hoặc đa kênh | Đầu ra là mảng trung bình 1 kênh `[1, 32000]` | **PASS** |
| **UT-03** | Resample đa tần số | File có SR = 44.1kHz, 48kHz, 22.05kHz | Tín hiệu chuyển đổi mượt về đúng 16.000 Hz, không lệch pha | **PASS** |
| **UT-04** | Đệm tín hiệu ngắn (Short Pad) | Đoạn âm thanh $< 2.0\text{s}$ (ví dụ $1.0\text{s}$) | Đệm số 0 đối xứng 2 đầu về đúng $32.000\text{ samples}$ | **PASS** |
| **UT-05** | Cắt tín hiệu dài (Long Center-crop) | Đoạn âm thanh $> 2.0\text{s}$ (ví dụ $4.0\text{s}$) | Cắt giữ đoạn chính giữa $32.000\text{ samples}$ | **PASS** |
| **UT-06** | Chuẩn hóa biên độ đỉnh (Peak Scale) | Tín hiệu biên độ lớn hoặc nhỏ bất kỳ | Đỉnh biên độ được scale về $[-0.95, 0.95]$, không bị vỡ tiếng | **PASS** |
| **UT-07** | Trích xuất Log-Mel Spectrogram | Đoạn âm thanh chuẩn $32.000\text{ samples}$ | Tensor đầu ra có shape `[1, 64, 63]`, dải giá trị $[0, 1]$ | **PASS** |
| **UT-08** | Tính xác định (Determinism) | Cùng một tín hiệu âm thanh xử lý 2 lần | Đầu ra tensor trùng khớp $100\%$ (`torch.allclose = True`) | **PASS** |

---

### 2.2. Cấp độ 2: Thu Nhận Âm Thanh & Luồng Thời Gian Thực (Audio Capture & Streaming - IT-01 -> IT-04)

| Mã test | Tên ca kiểm thử | Mô tả & Đầu vào | Kết quả kỳ vọng | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **IT-01** | Quét thiết bị Microphone phần cứng | Gọi `sounddevice.query_devices()` | Liệt kê đầy đủ mic, phát hiện thiết bị mặc định hợp lệ | **PASS** |
| **IT-02** | Thu âm đoạn mẫu thời gian thực | Thu âm $2.0\text{s}$ @ 16kHz mono từ mic | File WAV lưu thành công, RMS $> -60\text{ dBFS}$, không méo tiếng | **PASS** |
| **IT-03** | Xử lý khung trượt Ring Buffer | Luồng âm thanh liên tục trượt $0.5\text{s}$ | Bộ đệm rolling mượt mà, không tràn bộ nhớ, không trễ tích lũy | *Sắp tới (Tuần 5)* |
| **IT-04** | Fallback khi mất kết nối Mic | Ngắt kết nối mic đột ngột | Bắt exception an toàn, chuyển sang chế độ đợi hoặc file test | **PASS** |

---

### 2.3. Cấp độ 3: Toàn Vẹn Tập Dữ Liệu & Đóng Băng (Dataset Integrity - DT-01 -> DT-04)

| Mã test | Tên ca kiểm thử | Mô tả & Đầu vào | Kết quả kỳ vọng | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **DT-01** | Kiểm tra cấu trúc 9 thư mục sự kiện | Quét thư mục `data/raw_categorized/` | Đủ 9 thư mục, mỗi thư mục có $\ge 150$ clip, tổng 2.036 clip | **PASS** |
| **DT-02** | Kiểm tra chất lượng file âm thanh | Quét toàn bộ 2.036 tệp .wav | 100% file đọc được, 0 file hỏng, 0 file chứa NaN/Inf | **PASS** |
| **DT-03** | Kiểm tra chống rò rỉ (Zero Leakage) | So khớp `recording_id` giữa Train, Val, Test | Giao thoa Train $\cap$ Test = 0, Train $\cap$ Val = 0, Val $\cap$ Test = 0 | **PASS** |
| **DT-04** | Đóng băng dữ liệu qua Checksum MD5 | Đối soát `dataset_v1_checksums.md5` | 100% tệp khớp mã băm gốc, sẵn sàng huấn luyện | **PASS** |

---

### 2.4. Cấp độ 4: Hiệu Năng Mô Hình & Hệ Thống (Model & System Acceptance - MA-01 -> MA-05)

| Mã test | Tiêu chí đánh giá | Ngưỡng chấp nhận (Target) | Phương pháp đo | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **MA-01** | Tốc độ trích xuất Mel trên CPU | $\le 30.0\text{ ms / sample}$ | Đo thời gian bằng `time.perf_counter()` (Thực tế đạt **16.23 ms**) | **PASS** |
| **MA-02** | Độ trễ suy luận Custom 2D-CNN | $\le 40.0\text{ ms / khung}$ | Đo qua 100 lần suy luận liên tiếp trên CPU | *Tuần 3* |
| **MA-03** | Điểm Macro-F1 toàn hệ thống | $\ge 0.80$ (Model v2) | Đánh giá trên tập Test độc lập 299 clip | *Tuần 4* |
| **MA-04** | Recall các lớp khẩn cấp (`scream`, `gunshot`, `alarm`) | $\ge 0.85$ | Per-class Recall trên tập Test | *Tuần 4* |
| **MA-05** | Tỷ lệ cảnh báo giả (FAR) trên âm thanh nền | $\le 5.0\%$ | Đánh giá trên tập `background_normal` | *Tuần 4* |

---

## 3. Mẫu Biên Bản Nghiệm Thu Kỹ Thuật (Sign-off Template)

```text
================================================================================
                    BIÊN BẢN NGHIỆM THU KỸ THUẬT (MILESTONE SIGN-OFF)
================================================================================
Tên mốc nghiệm thu : [M1 / M2 / M3 / M4 / M5 / M6 / M7 / M8]
Giai đoạn thực hiện: Tuần ..... (Từ ngày .../.../2026 đến ngày .../.../2026)
Người chủ trì      : ...........................................................
Thành viên tham dự : TV1 (AI Lead), TV2 (Audio Lead), TV3 (QA Lead)

I. DANH SÁCH KẾT QUẢ ĐẦU RA ĐÃ KIỂM TRA
1. File mã nguồn / Cấu hình: ...................................................
2. File báo cáo / Hình ảnh : ...................................................
3. Bộ kiểm thử tự động     : ... / ... tests passed (100%)

II. KẾT QUẢ ĐỐI SOÁT VỚI TIÊU CHÍ CHẤP NHẬN
- Tiêu chí 1: [ĐẠT / CHƯA ĐẠT] - Ghi chú: ......................................
- Tiêu chí 2: [ĐẠT / CHƯA ĐẠT] - Ghi chú: ......................................
- Tiêu chí 3: [ĐẠT / CHƯA ĐẠT] - Ghi chú: ......................................

III. KẾT LUẬN & ĐÓNG BĂNG MỐC
[ ] ĐỒNG Ý NGHIỆM THU: Cho phép merge vào nhánh develop và chuyển sang tuần tiếp theo.
[ ] YÊU CẦU BỔ SUNG: Cần khắc phục các điểm tồn đọng trước khi họp lại.

Chữ ký đại diện 3 thành viên:
TV1: .....................    TV2: .....................    TV3: .....................
================================================================================
```
