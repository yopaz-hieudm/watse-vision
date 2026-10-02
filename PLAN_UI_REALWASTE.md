# Kế hoạch triển khai giao diện RealWaste

## 1. Mục tiêu

Xây dựng giao diện web Streamlit bằng tiếng Việt để người dùng có thể vận hành và theo dõi toàn bộ pipeline RealWaste mà không phải nhớ lệnh terminal.

Giao diện phải hỗ trợ:

- Xem trạng thái dữ liệu, EDA và train/validation/test split.
- Chọn một trong bốn thí nghiệm E1–E4 và bắt đầu huấn luyện.
- Theo dõi epoch, loss, Accuracy, Macro-F1, learning rate và log trong lúc train.
- Dừng an toàn hoặc tiếp tục một lần train từ `last.pt`.
- Chạy đánh giá checkpoint trên Validation hoặc Test.
- Xem metrics, confusion matrix, đường cong huấn luyện và các cặp lớp nhầm lẫn.
- So sánh tự động E1–E4.
- Tải ảnh lên và dự đoán bằng checkpoint đã chọn.
- Hiển thị rõ các điều kiện còn thiếu thay vì làm ứng dụng crash.

Phần train vẫn chạy trong một tiến trình nền. Streamlit chỉ điều khiển, đọc trạng thái và trình bày kết quả.

## 2. Phạm vi phiên bản đầu

### Có trong phạm vi

- Chạy cục bộ trên một máy.
- Một job nặng tại một thời điểm: download, EDA, split, train, evaluate, compare hoặc Grad-CAM.
- Bốn cấu hình cố định:
  - E1: MobileNetV3 Small, không augmentation.
  - E2: ResNet18, không augmentation.
  - E3: MobileNetV3 Small, có augmentation.
  - E4: ResNet18, có augmentation.
- Theo dõi job sau mỗi lần Streamlit rerun hoặc tải lại trang.
- CPU và CUDA thông qua `device: auto`.

### Chưa làm trong phiên bản đầu

- Đăng nhập và phân quyền nhiều người dùng.
- Chạy nhiều thí nghiệm song song.
- Huấn luyện phân tán hoặc điều khiển máy từ xa.
- Chỉnh tùy ý mọi hyperparameter trên giao diện.
- Deploy public lên Internet.

## 3. Kiến trúc được chọn

```text
Streamlit UI
    |
    +-- đọc JSON/CSV/PNG từ data/ và outputs/
    |
    +-- JobManager
            |
            +-- subprocess không dùng shell=True
            +-- chỉ chạy command thuộc allowlist
            +-- ghi PID, trạng thái và log
            |
            +-- scripts/train.py / evaluate.py / ...
                    |
                    +-- ghi progress.json nguyên tử
                    +-- ghi checkpoint và kết quả hiện có
```

Không gọi trực tiếp vòng lặp train trong request của Streamlit vì một lần rerun có thể làm mất trạng thái giao diện hoặc khóa toàn bộ UI.

### Trạng thái job

Mỗi job có thư mục:

```text
outputs/jobs/<job_id>/
├── job.json
├── progress.json
├── stdout.log
└── stderr.log
```

`job.json` chứa tối thiểu:

```json
{
  "job_id": "20261001_153000_E1_train",
  "kind": "train",
  "experiment_id": "E1",
  "command": ["python", "scripts/train.py", "--config", "..."],
  "pid": 12345,
  "status": "running",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "exit_code": null
}
```

Trạng thái hợp lệ: `queued`, `running`, `succeeded`, `failed`, `stopping`, `stopped`.

Ghi JSON bằng file tạm rồi `replace` để Streamlit không đọc trúng file đang ghi dở.

## 4. Cấu trúc file dự kiến

```text
app/
├── streamlit_app.py
├── components/
│   ├── __init__.py
│   ├── charts.py
│   ├── experiment_card.py
│   ├── job_status.py
│   └── layout.py
└── pages/
    ├── 1_Du_lieu.py
    ├── 2_Huan_luyen.py
    ├── 3_Danh_gia.py
    ├── 4_So_sanh.py
    └── 5_Du_doan.py

src/realwaste/ui/
├── __init__.py
├── artifact_reader.py
├── experiment_registry.py
├── job_manager.py
└── view_models.py

src/realwaste/training/
└── progress.py

scripts/
└── run_job.py

tests/
├── test_artifact_reader.py
├── test_experiment_registry.py
├── test_job_manager.py
└── test_training_progress.py
```

## 5. Các trang giao diện

### 5.1. Tổng quan — `app/streamlit_app.py`

Mục đích: cho biết dự án đã sẵn sàng đến đâu.

Hiển thị:

- Tên dự án và mô tả ngắn.
- Thiết bị hiện có: CPU/CUDA và tên GPU nếu có.
- Bốn thẻ trạng thái:
  - Dataset: số ảnh hợp lệ và số ảnh lỗi.
  - Split: số mẫu train/validation/test và fingerprint rút gọn.
  - Experiments: số E1–E4 đã có `best.pt`.
  - Evaluation: số experiment đã có `metrics.json`.
- Job đang chạy gần nhất.
- Bảng trạng thái E1–E4 gồm model, augmentation, checkpoint, metrics và hành động tiếp theo.
- Cảnh báo rõ nếu thiếu dữ liệu, split hoặc checkpoint.

### 5.2. Dữ liệu — `app/pages/1_Du_lieu.py`

Hiển thị:

- SHA-256 archive và trạng thái xác minh.
- Tổng số ảnh, số lớp, ảnh hợp lệ và ảnh lỗi.
- `class_distribution.png`.
- `sample_grid.png`.
- Bảng phân bố lớp từ `class_distribution.csv`.
- Bảng ảnh lỗi từ `invalid_images.csv`.
- Thông tin split và fingerprint.

Hành động:

- `Kiểm tra dữ liệu` chạy `scripts/download_data.py --verify-only`.
- `Tạo lại EDA` chạy `scripts/inspect_data.py`.
- `Tạo lại split` yêu cầu checkbox xác nhận vì sẽ thay đổi file split nếu seed/ratio đổi.

Phiên bản đầu không cho thay URL tải dữ liệu từ ô nhập tự do.

### 5.3. Huấn luyện — `app/pages/2_Huan_luyen.py`

Thành phần:

- Selectbox chọn E1, E2, E3 hoặc E4 từ registry cố định.
- Hiển thị config ở chế độ chỉ đọc:
  - model;
  - augmentation;
  - epochs;
  - batch size;
  - optimizer;
  - loss;
  - scheduler;
  - device;
  - tiêu chí checkpoint.
- Kiểm tra trước khi chạy:
  - dataset tồn tại;
  - split metadata tồn tại;
  - config hợp lệ;
  - không có job nặng khác đang chạy;
  - output directory không xung đột ngoài experiment đã chọn.

Nút hành động:

- `Bắt đầu huấn luyện`.
- `Tiếp tục từ last.pt`, chỉ bật khi checkpoint hợp lệ và split fingerprint khớp.
- `Dừng sau batch hiện tại` hoặc terminate mềm bằng `SIGTERM`.

Theo dõi:

- Progress bar `epoch hiện tại / tổng epoch`.
- Metric cards: train loss, validation loss, validation accuracy, validation Macro-F1 và learning rate.
- Biểu đồ cập nhật từ `history.csv`.
- Log cuối cùng, mặc định 200 dòng.
- Thời gian bắt đầu, thời gian chạy và PID.

Trang tự cập nhật mỗi 2 giây bằng `st.fragment(run_every="2s")`; không reload toàn bộ ứng dụng.

### 5.4. Đánh giá — `app/pages/3_Danh_gia.py`

- Chọn experiment đã có `best.pt` hoặc `last.pt`.
- Chọn split `validation` hoặc `test`; mặc định `test`.
- Nút `Chạy đánh giá`.
- Hiển thị Accuracy, Macro Precision, Macro Recall, Macro-F1 và loss.
- Hiển thị confusion matrix thường và normalized.
- Hiển thị classification report theo lớp.
- Hiển thị `top_confusions.csv`.
- Hiển thị training curves nếu có `history.csv`.
- Cho tải xuống `metrics.json`, `classification_report.csv` và `predictions.csv`.

Không ghi đè kết quả Test bằng kết quả Validation. Kết quả được tách thành:

```text
outputs/experiments/<experiment>/evaluation/validation/
outputs/experiments/<experiment>/evaluation/test/
```

Script `evaluate.py` cần nhận thêm `--output-dir` để hỗ trợ cấu trúc này.

### 5.5. So sánh — `app/pages/4_So_sanh.py`

- Bảng trạng thái E1–E4, nêu experiment nào còn thiếu kết quả Test.
- Nút `Tạo lại bảng so sánh`, chỉ bật khi đủ bốn `metrics.json` Test và cùng split fingerprint.
- Bảng xếp hạng theo Test Macro-F1.
- Biểu đồ Accuracy và Macro-F1.
- Bảng ảnh hưởng augmentation theo từng model.
- Đánh dấu model tốt nhất nhưng không tự tuyên bố augmentation tốt hơn nếu delta âm.
- Nút tải `experiment_summary.csv` và `augmentation_impact.csv`.

### 5.6. Dự đoán — `app/pages/5_Du_doan.py`

- Tự tìm tất cả `best.pt`/`last.pt`; không bắt người dùng gõ đường dẫn.
- Mặc định chọn checkpoint có Test Macro-F1 cao nhất; nếu chưa có metrics thì chọn `best.pt` mới nhất.
- Upload JPG/JPEG/PNG/WEBP.
- Xem trước ảnh.
- Chọn Top-K từ 1 đến 5, mặc định 3.
- Hiển thị:
  - lớp dự đoán;
  - confidence;
  - biểu đồ xác suất Top-K;
  - thời gian suy luận;
  - model và checkpoint đang dùng.
- Cache `Predictor` bằng `st.cache_resource`, key gồm đường dẫn checkpoint và thời gian sửa file.
- Xóa file tạm ngay sau dự đoán bằng `TemporaryDirectory`.
- Bắt và hiển thị lỗi ảnh hỏng, checkpoint sai hoặc thiếu RAM bằng thông báo dễ hiểu.

Grad-CAM cho một ảnh upload là hạng mục mở rộng sau khi luồng dự đoán cơ bản ổn định.

## 6. Experiment registry

Tạo registry cố định thay vì nhận command hoặc đường dẫn tùy ý từ UI:

```python
EXPERIMENTS = {
    "E1": "configs/mobilenet_v3_small_baseline.yaml",
    "E2": "configs/resnet18_baseline.yaml",
    "E3": "configs/mobilenet_v3_small_aug.yaml",
    "E4": "configs/resnet18_aug.yaml",
}
```

Registry chịu trách nhiệm:

- Load và validate config.
- Kiểm tra `experiment_id` trong YAML trùng với key.
- Resolve path bên trong project root.
- Tìm output directory, checkpoint, history và metrics.
- Không cho path traversal ra ngoài project.

## 7. Job manager và an toàn tiến trình

`JobManager` cung cấp:

```python
start_job(kind, experiment_id=None, args=None) -> JobRecord
get_job(job_id) -> JobRecord
get_active_job() -> JobRecord | None
list_jobs(limit=50) -> list[JobRecord]
stop_job(job_id) -> JobRecord
```

Quy định bắt buộc:

- Command được tạo từ allowlist; không nối chuỗi lệnh từ input người dùng.
- Dùng `subprocess.Popen([...], shell=False)`.
- Dùng `sys.executable` để job chạy đúng virtual environment của Streamlit.
- `cwd` luôn là project root đã resolve.
- Chỉ một job nặng có trạng thái `queued`, `running` hoặc `stopping`.
- Dùng lock file để tránh hai session cùng bấm chạy.
- Thêm `psutil` vào extra `app` để kiểm tra PID, command line và terminate cả process tree.
- Không kill PID nếu command/cwd không khớp metadata, tránh PID reuse.
- Stop gửi terminate mềm trước; sau timeout 10 giây mới cho phép force kill với xác nhận lần hai.
- stdout/stderr ghi vào file, không giữ pipe đầy gây deadlock.
- Khi app khởi động lại, trạng thái được phục hồi từ metadata và process thật.

## 8. Báo cáo tiến độ huấn luyện

Thêm `ProgressReporter` vào trainer. Sau mỗi epoch ghi:

```json
{
  "epoch": 4,
  "total_epochs": 30,
  "train_loss": 0.82,
  "train_accuracy": 0.71,
  "train_macro_f1": 0.69,
  "val_loss": 0.76,
  "val_accuracy": 0.74,
  "val_macro_f1": 0.72,
  "learning_rate": 0.001,
  "best_metric": 0.72,
  "updated_at": "..."
}
```

Không parse progress từ text log vì format log có thể thay đổi.

Trainer nhận callback tùy chọn để test được mà không phụ thuộc Streamlit. CLI cũ vẫn hoạt động khi không truyền callback.

## 9. Trải nghiệm và giao diện

- Dùng layout `wide`.
- Sidebar chỉ dùng để điều hướng và hiển thị job đang chạy.
- Màu sắc nhất quán: xanh lá cho thành công, vàng cho cảnh báo, đỏ cho lỗi.
- Không dựa riêng vào màu; luôn kèm icon và text trạng thái.
- Số metric hiển thị tối đa bốn chữ số thập phân.
- Các thao tác lâu có mô tả thời gian dự kiến, đặc biệt khi chỉ có CPU.
- Nút nguy hiểm đặt riêng và yêu cầu xác nhận.
- Không hiển thị traceback mặc định; traceback đầy đủ nằm trong log có thể mở rộng.
- Hỗ trợ màn hình laptop; bảng rộng dùng container width và cuộn ngang.

## 10. Thay đổi dependency và cấu hình

Trong `pyproject.toml`:

```toml
[project.optional-dependencies]
app = [
  "streamlit>=1.38,<2",
  "psutil>=6,<8",
]
```

Lệnh cài và chạy:

```bash
uv sync --extra dev --extra app
uv run streamlit run app/streamlit_app.py
```

Không thêm framework backend riêng ở phiên bản đầu.

## 11. Kiểm thử

### Unit test

- Registry chỉ chấp nhận E1–E4 và từ chối path traversal.
- Artifact reader xử lý file thiếu, JSON lỗi và CSV rỗng.
- Job manager tạo đúng command, không dùng shell và chặn job thứ hai.
- Job manager nhận biết process hoàn tất/thất bại.
- Stop chỉ terminate process thuộc đúng job.
- Progress reporter ghi JSON nguyên tử và đúng schema.
- Predictor cache invalidated khi checkpoint thay đổi.

### Integration test

- Chạy job giả ngắn và theo dõi từ `queued` đến `succeeded`.
- Chạy job giả lỗi và kiểm tra `stderr.log`, exit code và trạng thái `failed`.
- Resume đọc đúng `last.pt` và split fingerprint.
- Evaluate ghi riêng Validation/Test.
- Compare chỉ chạy khi đủ E1–E4 Test metrics.

### Smoke test UI

- App khởi động không cần checkpoint.
- Tất cả trang mở được khi thiếu artefact.
- Trang dữ liệu đọc đúng dataset hiện tại.
- Bấm train tạo đúng một job.
- Reload trang vẫn thấy job.
- Upload ảnh hợp lệ dự đoán được khi có checkpoint.
- Upload file hỏng hiển thị lỗi và không làm app crash.

## 12. Thứ tự triển khai

### Giai đoạn 1 — Nền tảng

1. Thêm dependency app và cập nhật lock files.
2. Tạo experiment registry và artifact reader.
3. Tạo job schema, atomic JSON helpers và job manager.
4. Viết unit test cho các module nền tảng.

### Giai đoạn 2 — Theo dõi huấn luyện

1. Thêm progress reporter vào trainer.
2. Thêm wrapper `run_job.py` để cập nhật trạng thái kết thúc.
3. Kiểm thử start, stop, fail và restore job.

### Giai đoạn 3 — UI đọc dữ liệu

1. Xây trang Tổng quan.
2. Xây trang Dữ liệu.
3. Xây component trạng thái và biểu đồ dùng chung.

### Giai đoạn 4 — UI vận hành model

1. Xây trang Huấn luyện.
2. Sửa evaluate để tách output Validation/Test.
3. Xây trang Đánh giá và So sánh.
4. Nâng cấp trang Dự đoán.

### Giai đoạn 5 — Hoàn thiện

1. Bổ sung xử lý lỗi và cảnh báo CPU.
2. Chạy lint, unit test, integration test và smoke test UI.
3. Cập nhật README với ảnh chụp và hướng dẫn sử dụng.

## 13. Tiêu chí hoàn thành

Giao diện được xem là hoàn thành khi:

- Khởi động bằng một lệnh Streamlit và không lỗi khi chưa có checkpoint.
- Hiển thị đúng EDA và split hiện tại.
- Có thể bắt đầu đúng E1–E4 mà không nhập command thủ công.
- UI không bị khóa trong lúc train.
- Tiến độ và log tiếp tục hiển thị sau khi reload trang.
- Không thể khởi động hai job nặng đồng thời.
- Có thể dừng an toàn và resume từ `last.pt`.
- Evaluation tách riêng Validation/Test và hiển thị đầy đủ metric/figure.
- Compare chỉ dùng Test metrics cùng split fingerprint.
- Dự đoán ảnh hoạt động với checkpoint hợp lệ và xử lý ảnh lỗi an toàn.
- Không dùng `shell=True`, không cho chạy command/path tùy ý.
- Toàn bộ test mới và test hiện tại đều pass; Ruff sạch.

## 14. Kết quả bàn giao dự kiến

- Giao diện Streamlit đa trang bằng tiếng Việt.
- Job manager chạy nền và lưu trạng thái bền vững.
- Dashboard theo dõi train/evaluate.
- Trang trực quan hóa EDA và so sánh E1–E4.
- Trang dự đoán hoàn chỉnh.
- Test cho logic UI/backend, README và lệnh khởi chạy.

