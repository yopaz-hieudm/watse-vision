# Kế hoạch code dự án phân loại rác thải RealWaste

## 1. Mục tiêu kỹ thuật

Xây dựng một pipeline phân loại ảnh có thể chạy lại và so sánh công bằng hai mô hình MobileNetV3 và ResNet18 trên bộ dữ liệu RealWaste.

Pipeline phải hỗ trợ:

- Khảo sát và kiểm tra dữ liệu 9 lớp.
- Chia Train, Validation và Test theo lớp.
- Huấn luyện bằng Transfer Learning.
- Bật hoặc tắt Data Augmentation bằng cấu hình.
- Lưu checkpoint và lịch sử huấn luyện.
- Đánh giá Accuracy, Precision, Recall và Macro-F1.
- Sinh Confusion Matrix, classification report và Grad-CAM.
- Chạy dự đoán một ảnh.
- Mở rộng thành demo Streamlit khi phần cốt lõi đã ổn định.

## 2. Công nghệ đề xuất

- Python 3.11.
- PyTorch và Torchvision.
- scikit-learn cho metric và chia dữ liệu.
- pandas và NumPy cho dữ liệu kết quả.
- Matplotlib và Seaborn cho biểu đồ.
- PyYAML cho file cấu hình.
- TensorBoard hoặc CSV log cho lịch sử huấn luyện.
- Streamlit cho demo tùy chọn.
- pytest cho unit test.
- Ruff cho lint và format.

## 3. Cấu trúc thư mục

```text
waste-vision/
├── configs/
│   ├── mobilenet_v3_small_baseline.yaml
│   ├── mobilenet_v3_small_aug.yaml
│   ├── resnet18_baseline.yaml
│   └── resnet18_aug.yaml
├── data/
│   ├── checksums/
│   │   └── realwaste.sha256
│   ├── downloads/
│   │   └── realwaste.zip
│   ├── raw/
│   │   └── RealWaste/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── realwaste/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── seed.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── split.py
│       │   ├── transforms.py
│       │   └── dataloader.py
│       ├── models/
│       │   ├── factory.py
│       │   ├── mobilenet_v3.py
│       │   └── resnet18.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   └── checkpoint.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── confusion_matrix.py
│       │   ├── error_analysis.py
│       │   └── gradcam.py
│       └── inference/
│           └── predictor.py
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── create_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_experiments.py
│   ├── generate_gradcam.py
│   └── predict.py
├── tests/
│   ├── test_config.py
│   ├── test_download.py
│   ├── test_eda.py
│   ├── test_split.py
│   ├── test_transforms.py
│   ├── test_model_factory.py
│   ├── test_metrics.py
│   ├── test_compare_experiments.py
│   └── test_predictor.py
├── outputs/
│   ├── eda/
│   ├── experiments/
│   └── comparison/
├── app/
│   └── streamlit_app.py
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

Không commit dữ liệu gốc, checkpoint lớn hoặc toàn bộ thư mục `outputs`. Chỉ commit file split, cấu hình và các kết quả nhỏ cần cho báo cáo.

## 4. Quy ước dữ liệu

### 4.1 Tải và xác minh dữ liệu

Nguồn duy nhất được dùng là UCI Machine Learning Repository:

```text
Dataset ID: 908
Page: https://archive.ics.uci.edu/dataset/908/realwaste
Download: https://archive.ics.uci.edu/static/public/908/realwaste.zip
```

File `scripts/download_data.py` phải:

1. Tải file vào `data/downloads/realwaste.zip.part`.
2. Đổi tên thành `realwaste.zip` chỉ khi tải hoàn tất.
3. Tính SHA-256 theo kiểu streaming để không nạp toàn bộ file vào RAM.
4. So sánh với giá trị đã ghim trong `data/checksums/realwaste.sha256`.
5. Dừng và xóa file `.part` nếu checksum không khớp.
6. Giải nén an toàn vào thư mục tạm, chặn đường dẫn ZIP traversal.
7. Chỉ chuyển dữ liệu hợp lệ vào `data/raw/RealWaste` sau khi xác nhận đủ 9 thư mục lớp.
8. Không tải hoặc giải nén lại nếu dữ liệu hiện tại đã vượt qua bước xác minh.

Checksum được lấy một lần từ bản tải tin cậy trên trang UCI, sau đó commit file `realwaste.sha256` vào repository. Không tự động chấp nhận checksum của một file vừa tải từ nguồn chưa được xác minh.

Lệnh mục tiêu:

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

Không commit `realwaste.zip` hoặc thư mục ảnh vào Git.

### 4.2 Nhãn chuẩn

```python
CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]
```

Thứ tự nhãn phải được lưu trong checkpoint và dùng lại khi đánh giá hoặc dự đoán.

### 4.3 File split

Mỗi file CSV nên có cấu trúc:

```text
path,label,class_name
raw/Cardboard/image_001.jpg,0,Cardboard
```

Yêu cầu:

- Chia theo lớp bằng stratified split với tỷ lệ Train 70%, Validation 15% và Test 15%.
- Dùng `seed=42` cho bước chia dữ liệu.
- Tạo split một lần rồi lưu ra file.
- Kiểm tra không có đường dẫn trùng giữa ba tập.
- Không tạo lại split trong từng lần huấn luyện.

### 4.4 Transform

Validation và Test chỉ sử dụng transform xác định:

1. Resize cạnh ngắn về 256 và center crop 224 x 224.
2. Chuyển thành tensor.
3. Normalize theo ImageNet.

Train có hai chế độ:

- `baseline`: giống Validation và Test.
- `augmentation`: dùng `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)` và `ColorJitter` mức nhẹ trước bước normalize.

Normalization dùng đúng thống kê ImageNet của pretrained weights. Không dùng augmentation ngẫu nhiên cho Validation hoặc Test.

## 5. Thiết kế cấu hình

Mọi tham số huấn luyện phải lấy từ YAML, không hard-code trong script.

```yaml
experiment_id: E3
experiment_name: mobilenet_v3_small_aug
seed: 42
device: auto

data:
  root: data/raw/RealWaste
  train_csv: data/splits/train.csv
  val_csv: data/splits/val.csv
  test_csv: data/splits/test.csv
  image_size: 224
  resize_size: 256
  batch_size: 32
  num_workers: 4
  pin_memory: true
  split:
    train: 0.70
    validation: 0.15
    test: 0.15
  augmentation:
    enabled: true
    random_resized_crop_scale: [0.8, 1.0]
    horizontal_flip_probability: 0.5
    rotation_degrees: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.05

model:
  name: mobilenet_v3_small
  weights: IMAGENET1K_V1
  pretrained: true
  num_classes: 9
  train_backbone: false

training:
  epochs: 30
  mixed_precision: true
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
  loss:
    name: cross_entropy
    label_smoothing: 0.0
  scheduler:
    name: reduce_lr_on_plateau
    monitor: val_macro_f1
    mode: max
    factor: 0.5
    patience: 2
    min_learning_rate: 0.000001
  early_stopping:
    monitor: val_macro_f1
    mode: max
    patience: 5
    minimum_delta: 0.0001

checkpoint:
  monitor: val_macro_f1
  mode: max
  save_best: true
  save_last: true

output:
  directory: outputs/experiments/mobilenet_v3_small_aug
```

Đây là cấu hình chuẩn cho E3. E1 dùng cùng cấu hình nhưng đặt `augmentation.enabled: false`. E2 và E4 chỉ đổi `model.name` thành `resnet18`. Batch size giảm từ 32 xuống 16 hoặc 8 nếu thiếu bộ nhớ; mọi thay đổi phải được lưu trong bản sao YAML của lần chạy.

`device: auto` chọn CUDA trước, sau đó MPS nếu khả dụng, cuối cùng mới dùng CPU. Mixed precision chỉ được bật trên CUDA; code phải tự tắt tính năng này trên CPU hoặc backend không hỗ trợ.

Bốn thí nghiệm E1 đến E4 dùng `CrossEntropyLoss` không có class weights để giữ phép so sánh chính nhất quán. Class weighting chỉ là thí nghiệm mở rộng sau khi hoàn thành E1 đến E4. Fine-tuning dùng learning rate `0.0001` và phải lưu thành file cấu hình riêng.

## 6. Module cần triển khai

### 6.1 Đọc cấu hình

File: `src/realwaste/config.py`

Trách nhiệm:

- Đọc YAML.
- Kiểm tra trường bắt buộc.
- Tạo đường dẫn đầu ra.
- Lưu bản sao cấu hình vào thư mục kết quả.

Điều kiện hoàn thành:

- Báo lỗi rõ khi thiếu khóa cấu hình.
- Không ghi đè kết quả cũ nếu chưa được cho phép.

### 6.2 Tải và xác minh dữ liệu

Files:

- `scripts/download_data.py`
- `data/checksums/realwaste.sha256`

Trách nhiệm:

- Tải đúng file ZIP từ UCI Dataset ID 908.
- Hỗ trợ tiếp tục hoặc tải lại an toàn khi kết nối bị gián đoạn.
- Xác minh SHA-256 trước khi giải nén.
- Chặn ZIP traversal khi giải nén.
- Xác nhận đủ 9 thư mục lớp và đúng 4.752 ảnh hợp lệ trước khi báo thành công.
- Trả mã lỗi khác 0 khi URL, checksum, cấu trúc thư mục hoặc số ảnh không hợp lệ.

### 6.3 Cố định random seed

File: `src/realwaste/seed.py`

Trách nhiệm:

- Cố định seed cho Python, NumPy và PyTorch.
- Cấu hình CUDA deterministic khi cần tái lập nghiêm ngặt.
- Ghi seed vào log và checkpoint.

### 6.4 Khảo sát và chia dữ liệu

Files:

- `src/realwaste/data/split.py`
- `scripts/inspect_data.py`
- `scripts/create_splits.py`

Trách nhiệm:

- Quét ảnh và ánh xạ nhãn.
- Thử mở toàn bộ ảnh và ghi ảnh lỗi.
- Thống kê số ảnh theo lớp.
- Tạo lưới ảnh mẫu gồm 4 ảnh cho mỗi lớp.
- Tạo Train, Validation và Test theo lớp với tỷ lệ 70/15/15 và seed 42.
- Tính split fingerprint từ nội dung ba file CSV.
- Xuất ba file CSV và báo cáo kiểm tra trùng lặp.

Đầu ra EDA bắt buộc:

- `outputs/eda/class_distribution.csv`.
- `outputs/eda/class_distribution.png`.
- `outputs/eda/sample_grid.png`.
- `outputs/eda/invalid_images.csv`, có header kể cả khi không có ảnh lỗi.
- `outputs/eda/dataset_summary.json`.

`dataset_summary.json` tối thiểu chứa tổng số file, số ảnh hợp lệ, số ảnh lỗi, số lớp, số ảnh từng lớp và thời điểm tạo báo cáo. Bước EDA chỉ đạt khi có đủ 9 lớp và tổng số ảnh hợp lệ là 4.752.

### 6.5 Dataset và DataLoader

Files:

- `src/realwaste/data/dataset.py`
- `src/realwaste/data/transforms.py`
- `src/realwaste/data/dataloader.py`

Interface chính:

```python
def build_transforms(image_size: int, augmentation: bool): ...


def build_dataloaders(config):
    return train_loader, val_loader, test_loader
```

Điều kiện hoàn thành:

- Batch ảnh có shape `[B, 3, 224, 224]`.
- Nhãn nằm trong khoảng từ 0 đến 8.
- Validation và Test không chứa transform ngẫu nhiên.
- DataLoader chạy được với `num_workers=0` trên mọi máy.

### 6.6 Model factory

Files:

- `src/realwaste/models/factory.py`
- `src/realwaste/models/mobilenet_v3.py`
- `src/realwaste/models/resnet18.py`

Interface chính:

```python
def create_model(name: str, num_classes: int, pretrained: bool = True): ...


def set_backbone_trainable(model, mode: str): ...
```

Yêu cầu:

- Chỉ hỗ trợ `mobilenet_v3_small` cho nhánh MobileNetV3, dùng `MobileNet_V3_Small_Weights.IMAGENET1K_V1`.
- Hỗ trợ `resnet18` với `ResNet18_Weights.IMAGENET1K_V1`.
- Dùng API `weights=...` hiện hành của Torchvision, không dùng tham số `pretrained=True` đã cũ.
- Tham chiếu MobileNetV3 Small: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- Thay classification head thành 9 đầu ra.
- Hỗ trợ chế độ chỉ train head và fine-tune layer cuối.
- Lưu tên model và thứ tự nhãn trong checkpoint.

### 6.7 Trainer

Files:

- `src/realwaste/training/trainer.py`
- `src/realwaste/training/callbacks.py`
- `src/realwaste/training/checkpoint.py`

Trainer cần có:

- Vòng lặp train và validation theo epoch.
- Tính loss, accuracy và Macro-F1.
- Automatic Mixed Precision khi dùng CUDA.
- Learning-rate scheduler nếu được bật trong cấu hình.
- Early stopping theo `val_macro_f1`.
- Lưu `best.pt` và `last.pt`.
- Xuất `history.csv`.
- Resume từ checkpoint.

Checkpoint tối thiểu chứa:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": ...,
    "best_metric": ...,
    "config": ...,
    "class_names": ...,
    "model_name": ...,
    "weights_id": ...,
    "split_fingerprint": ...,
}
```

### 6.8 Đánh giá

Files:

- `src/realwaste/evaluation/metrics.py`
- `src/realwaste/evaluation/confusion_matrix.py`
- `scripts/evaluate.py`

Đầu ra bắt buộc:

- `metrics.json` gồm Accuracy, Macro Precision, Macro Recall, Macro-F1, loss, số mẫu, checkpoint, model name và split fingerprint.
- `classification_report.csv`.
- `predictions.csv` gồm đường dẫn, nhãn thật, nhãn dự đoán và xác suất.
- `confusion_matrix.png`.
- Curves của loss, accuracy và Macro-F1.

Không chạy đánh giá Test trong quá trình chọn hyperparameter.

### 6.9 So sánh thí nghiệm

File: `scripts/compare_experiments.py`

Trách nhiệm:

- Chỉ đọc kết quả của E1, E2, E3 và E4 khi cả bốn lần chạy đã hoàn tất.
- Kiểm tra bốn thí nghiệm dùng cùng split fingerprint, seed và tập Test.
- Đọc `metrics.json`, `history.csv` và bản sao cấu hình của từng lần chạy.
- Tạo `outputs/comparison/experiment_summary.csv`.
- Tạo `outputs/comparison/experiment_summary.md` để chèn vào báo cáo.
- Tạo `outputs/comparison/metrics_comparison.png`.
- Tạo `outputs/comparison/augmentation_impact.csv`.
- Xếp hạng mô hình chính theo Test Macro-F1; Accuracy chỉ là chỉ số bổ sung.

Các cột tối thiểu của `experiment_summary.csv`:

```text
experiment_id,model,augmentation,best_epoch,accuracy,macro_precision,macro_recall,macro_f1,checkpoint_path,split_fingerprint
```

Script phải dừng nếu thiếu thí nghiệm, thiếu metric bắt buộc hoặc phát hiện split fingerprint khác nhau.

### 6.10 Phân tích lỗi và Grad-CAM

Files:

- `src/realwaste/evaluation/error_analysis.py`
- `src/realwaste/evaluation/gradcam.py`
- `scripts/generate_gradcam.py`

Trách nhiệm:

- Tìm các cặp lớp bị nhầm nhiều nhất.
- Lấy ảnh đúng có độ tin cậy cao và thấp.
- Lấy ảnh sai có độ tin cậy cao.
- Sinh ảnh gốc, heatmap và ảnh overlay.
- Lưu metadata của từng ảnh Grad-CAM.

### 6.11 Predictor

Files:

- `src/realwaste/inference/predictor.py`
- `scripts/predict.py`

Interface đề xuất:

```python
class Predictor:
    def __init__(self, checkpoint_path: str, device: str = "auto"): ...

    def predict(self, image_path: str, top_k: int = 3) -> list[dict]: ...
```

Predictor phải lấy preprocessing và class names từ checkpoint hoặc cấu hình đi kèm.

## 7. Thứ tự triển khai

### Giai đoạn 1: Khởi tạo dự án

- Tạo cấu trúc thư mục.
- Tạo môi trường Python 3.11 và dependency.
- Thiết lập Ruff, pytest và `.gitignore`.
- Tạo bốn file cấu hình E1 đến E4.
- Tạo `requirements-lock.txt` sau khi kiểm thử dependency.

Hoàn thành khi:

- `pytest` chạy được.
- `python scripts/train.py --help` chạy được.
- README có hướng dẫn cài đặt cơ bản.
- Một môi trường mới cài được đúng dependency từ file lock.

### Giai đoạn 2: Tải, EDA và pipeline dữ liệu

- Viết và chạy `download_data.py`.
- Ghim checksum SHA-256 từ bản tải tin cậy của UCI.
- Chạy `inspect_data.py` và tạo đủ năm đầu ra EDA bắt buộc.
- Tạo split 70/15/15 với seed 42.
- Lưu split fingerprint.
- Viết Dataset, transforms và DataLoader.
- Viết test cho download, split và transforms.

Hoàn thành khi:

- Checksum ZIP hợp lệ.
- Có đúng 9 lớp và 4.752 ảnh hợp lệ.
- Có `class_distribution.csv`, `class_distribution.png`, `sample_grid.png`, `invalid_images.csv` và `dataset_summary.json`.
- Đọc được một batch của cả ba tập.
- Không có ảnh trùng giữa các tập.
- Augmentation chỉ xuất hiện ở Train.

### Giai đoạn 3: Baseline MobileNetV3 Small

- Viết model factory.
- Viết trainer tối thiểu.
- Huấn luyện thử 1 đến 2 epoch trên tập nhỏ.
- Huấn luyện đầy đủ E1.

Hoàn thành khi:

- Loss giảm trên tập nhỏ.
- Lưu và tải lại checkpoint được.
- Có `history.csv` và metric Validation.

### Giai đoạn 4: Baseline ResNet18

- Thêm ResNet18 vào model factory.
- Chạy cùng split và cấu hình đánh giá.
- Hoàn thành E2.

Hoàn thành khi:

- E1 và E2 có cùng schema kết quả.
- Có thể tạo một bảng so sánh tự động.

### Giai đoạn 5: Data Augmentation

- Chốt các transform augmentation.
- Chạy E3 và E4.
- So sánh từng mô hình trước và sau augmentation.

Hoàn thành khi:

- Có đủ bốn cấu hình bắt buộc.
- Mỗi cấu hình có checkpoint, log và metric Validation.

### Giai đoạn 6: Fine-tuning

- Chọn cấu hình tốt nhất bằng Validation.
- Mở một số layer cuối.
- Giảm learning rate.
- Chạy E5 và E6 nếu đủ tài nguyên.

Fine-tuning là phần có điều kiện. Không trì hoãn đánh giá Test chỉ để hoàn thành phần này.

### Giai đoạn 7: Đánh giá Test và so sánh E1 đến E4

- Khóa cấu hình và checkpoint.
- Chạy Test một lần cho E1, E2, E3 và E4.
- Xuất metric, report, prediction và Confusion Matrix của từng thí nghiệm.
- Chạy `compare_experiments.py`.
- Tạo bảng tổng hợp và biểu đồ so sánh tự động.

Hoàn thành khi:

- Có Accuracy, Macro Precision, Macro Recall và Macro-F1.
- Có metric theo từng lớp.
- Tất cả mô hình dùng cùng Test split và split fingerprint.
- Có `experiment_summary.csv`, `experiment_summary.md`, `metrics_comparison.png` và `augmentation_impact.csv`.

### Giai đoạn 8: Phân tích lỗi và Grad-CAM

- Xác định lớp và cặp lớp khó.
- Chọn ảnh đúng và sai đại diện.
- Sinh Grad-CAM và viết nhận xét.

Hoàn thành khi:

- Có ít nhất một trường hợp đúng và sai cho mô hình được chọn.
- Nhận xét phân biệt rõ đặc trưng vật thể và ảnh hưởng của nền.

### Giai đoạn 9: Đóng gói

- Hoàn thiện README.
- Tạo lệnh dự đoán một ảnh.
- Dọn code và chạy toàn bộ test.
- Đối chiếu số liệu với báo cáo.
- Làm Streamlit nếu còn thời gian.

## 8. Ma trận thí nghiệm

| Mã | Backbone | Augmentation | Chế độ | Ưu tiên |
|---|---|---:|---|---|
| E1 | MobileNetV3 Small | Không | Train head | Bắt buộc |
| E2 | ResNet18 | Không | Train head | Bắt buộc |
| E3 | MobileNetV3 Small | Có | Train head | Bắt buộc |
| E4 | ResNet18 | Có | Train head | Bắt buộc |
| E5 | Mô hình tốt hơn | Cấu hình tốt | Fine-tune layer cuối | Có điều kiện |
| E6 | Mô hình còn lại | Cấu hình tốt | Fine-tune layer cuối | Có điều kiện |

Mỗi lần chạy phải có một thư mục riêng:

```text
outputs/experiments/<experiment_name>/
├── config.yaml
├── best.pt
├── last.pt
├── history.csv
├── metrics.json
├── classification_report.csv
├── predictions.csv
└── figures/

outputs/comparison/
├── experiment_summary.csv
├── experiment_summary.md
├── metrics_comparison.png
└── augmentation_impact.csv
```

## 9. Unit test tối thiểu

### `test_download.py`

- Từ chối file có checksum sai.
- Không để lại file ZIP hoàn chỉnh khi tải bị gián đoạn.
- Chặn đường dẫn ZIP traversal.
- Không tải lại khi file và thư mục dữ liệu đã hợp lệ.

### `test_eda.py`

- Tạo đủ năm đầu ra EDA bắt buộc.
- Tổng số ảnh theo lớp bằng số ảnh hợp lệ.
- `invalid_images.csv` luôn có header.
- Báo lỗi khi thiếu lớp hoặc số ảnh hợp lệ không bằng 4.752.

### `test_split.py`

- Tổng số ảnh sau chia bằng tổng số ảnh hợp lệ.
- Không có đường dẫn trùng giữa Train, Validation và Test.
- Cả ba tập chứa đủ lớp nếu số lượng ảnh cho phép.
- Cùng seed tạo cùng kết quả.

### `test_transforms.py`

- Tensor đầu ra có 3 kênh.
- Kích thước đúng 224 x 224.
- Validation transform cho cùng kết quả với cùng ảnh.
- Train augmentation tạo được ảnh hợp lệ.

### `test_model_factory.py`

- Tạo được MobileNetV3 và ResNet18.
- Output có shape `[batch_size, 9]`.
- Chế độ train head đóng băng backbone.
- Chế độ fine-tune mở đúng layer dự kiến.

### `test_metrics.py`

- Metric khớp một bộ nhãn nhỏ tính bằng tay.
- Macro-F1 không bị thay bằng weighted F1.
- Confusion Matrix có shape 9 x 9.

### `test_compare_experiments.py`

- Từ chối so sánh khi thiếu E1, E2, E3 hoặc E4.
- Từ chối kết quả có split fingerprint khác nhau.
- Tạo đủ bốn file trong `outputs/comparison`.
- Xếp hạng theo Macro-F1 giảm dần.

### `test_predictor.py`

- Tải được checkpoint.
- Trả về `top_k` kết quả.
- Xác suất nằm trong khoảng từ 0 đến 1.
- Tổng xác suất của 9 lớp xấp xỉ 1.

## 10. Lệnh CLI mục tiêu

```bash
# Tải, xác minh checksum và giải nén dữ liệu
python scripts/download_data.py

# Chỉ kiểm tra lại ZIP và dữ liệu đã có
python scripts/download_data.py --verify-only

# Kiểm tra dữ liệu và tạo đầu ra EDA
python scripts/inspect_data.py --data-dir data/raw/RealWaste --output-dir outputs/eda

# Tạo split 70/15/15 cố định
python scripts/create_splits.py --data-dir data/raw/RealWaste --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42

# Huấn luyện E1
python scripts/train.py --config configs/mobilenet_v3_small_baseline.yaml

# Tiếp tục E1 từ checkpoint
python scripts/train.py --config configs/mobilenet_v3_small_baseline.yaml --resume outputs/experiments/mobilenet_v3_small_baseline/last.pt

# Đánh giá Test
python scripts/evaluate.py --checkpoint outputs/experiments/mobilenet_v3_small_baseline/best.pt --split test

# So sánh tự động E1 đến E4
python scripts/compare_experiments.py --input-dir outputs/experiments --output-dir outputs/comparison

# Sinh Grad-CAM
python scripts/generate_gradcam.py --checkpoint outputs/experiments/mobilenet_v3_small_aug/best.pt --samples 20

# Dự đoán một ảnh
python scripts/predict.py --checkpoint outputs/experiments/mobilenet_v3_small_aug/best.pt --image sample.jpg --top-k 3

# Chạy kiểm tra chất lượng
ruff check .
ruff format --check .
pytest -q
```

## 11. Definition of Done

Phần code được xem là hoàn thành khi:

- Cài đặt được theo README trên môi trường Python 3.11 mới bằng dependency đã khóa phiên bản.
- Tải được RealWaste từ UCI, xác minh đúng SHA-256 và kiểm tra đủ 9 lớp với 4.752 ảnh hợp lệ.
- Có đủ năm đầu ra EDA bắt buộc.
- Tạo lại đúng split 70/15/15 với seed 42 hoặc sử dụng ba file split đã lưu.
- Hoàn thành E1 đến E4 bằng MobileNetV3 Small và ResNet18 trên cùng split fingerprint.
- Checkpoint có thể tải lại để tiếp tục huấn luyện, đánh giá và dự đoán.
- Có đủ Accuracy, Macro Precision, Macro Recall, Macro-F1 và Confusion Matrix trên Test.
- Có classification report theo 9 lớp.
- Có đủ bốn đầu ra so sánh tự động E1 đến E4.
- Có phân tích lỗi và Grad-CAM cho cả dự đoán đúng và sai.
- Ruff và toàn bộ unit test chính chạy thành công.
- Không hard-code đường dẫn máy cá nhân.
- Báo cáo và slide sử dụng đúng số liệu được sinh từ thư mục `outputs`.

## 12. Việc nên code đầu tiên

1. Tạo `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`, `.gitignore` và cấu trúc package.
2. Viết `download_data.py`, ghim checksum và kiểm thử tải lại, checksum sai cùng ZIP traversal.
3. Viết `inspect_data.py` và tạo đủ năm đầu ra EDA bắt buộc.
4. Viết `create_splits.py` cho tỷ lệ 70/15/15, seed 42 và split fingerprint.
5. Viết Dataset, transforms và DataLoader.
6. Chạy smoke test đọc một batch của cả ba tập.
7. Viết model factory cho `mobilenet_v3_small` và ResNet18.
8. Viết trainer tối thiểu và overfit một tập 20 đến 50 ảnh.
9. Chạy E1 đến E4.
10. Viết `compare_experiments.py`, đánh giá Test và sinh bảng so sánh.
11. Chỉ sau đó mới triển khai fine-tuning, Grad-CAM và Streamlit.

## 13. Đợt hoàn thiện sau audit nghiệp vụ

Kết quả audit ngày 02/10/2026 xác nhận pipeline dữ liệu, EDA, split và E1 đã chạy
thành công, nhưng dự án chưa đạt Definition of Done vì E2-E4 và toàn bộ artefact
đánh giá Test/so sánh/Grad-CAM chưa được tạo. Trước khi chạy các thí nghiệm còn lại,
cần hoàn thiện các kiểm soát sau để tránh sinh kết quả sai hoặc ghi đè artefact.

### 13.1. An toàn huấn luyện và resume

- CLI train phải từ chối chạy mới nếu thư mục experiment đã có `history.csv`,
  `best.pt` hoặc `last.pt`, trừ khi người dùng truyền cờ ghi đè rõ ràng.
- `--resume` và chế độ ghi đè loại trừ lẫn nhau.
- Trước khi cho phép resume, cả UI và JobManager phải kiểm tra checkpoint có đủ metadata,
  đúng experiment/model và cùng split fingerprint hiện tại.
- Bổ sung integration test cho resume thành công và từ chối checkpoint khác fingerprint.

### 13.2. Đánh giá và so sánh

- Trang Đánh giá phải hiển thị `training_curves.png` khi artefact tồn tại.
- Trang So sánh chỉ bật nút tạo bảng khi đủ E1-E4 Test metrics và tất cả cùng split
  fingerprint; backend vẫn phải kiểm tra lại để không phụ thuộc UI.
- Bước compare phải đọc `config.yaml` và `history.csv` của từng experiment, xác nhận cùng
  seed, cùng Test CSV, metadata model/augmentation nhất quán và best epoch có trong history.

### 13.3. Job và trải nghiệm vận hành

- Theo dõi job phải hiển thị learning rate và thời gian đã chạy.
- Khi terminate mềm quá timeout, UI phải cung cấp nút force kill riêng với xác nhận lần hai.
- Không hiển thị traceback trực tiếp trên UI; lỗi chi tiết tiếp tục lưu trong log.

### 13.4. Dữ liệu và giải thích mô hình

- Không tải lại archive nếu dataset hiện tại đã vượt qua kiểm tra cấu trúc nghiêm ngặt.
- Grad-CAM phải chọn đủ ba nhóm: dự đoán đúng confidence cao, dự đoán đúng confidence
  thấp và dự đoán sai confidence cao; metadata phải ghi rõ nhóm của từng ảnh.
- Bổ sung test cho luồng không tải lại, lựa chọn mẫu Grad-CAM và các kiểm soát mới.

### 13.5. Thứ tự đóng các hạng mục còn lại

1. Hoàn thiện các kiểm soát và test ở mục 13.1-13.4; chạy Ruff và toàn bộ pytest.
2. Huấn luyện E2, E3 và E4 trên cùng split fingerprint với E1.
3. Đánh giá `best.pt` của E1-E4 trên Test đúng một lần.
4. Sinh đủ bốn artefact comparison và kiểm tra xếp hạng theo Test Macro-F1.
5. Sinh Grad-CAM cho mô hình được chọn, gồm cả trường hợp đúng và sai, rồi đối chiếu số liệu
   trong báo cáo/slide với `outputs`.


### 13.6. Trạng thái thực hiện

- [x] Bổ sung guard chống ghi đè và cờ `--overwrite`.
- [x] Xác thực checkpoint/fingerprint trước resume ở UI và JobManager.
- [x] Kiểm tra Test metrics/fingerprint ở UI; kiểm tra config, history, seed và Test CSV ở backend.
- [x] Hiển thị training curves, learning rate, thời gian chạy và force kill có xác nhận.
- [x] Không tải lại dataset hợp lệ và chọn đủ ba nhóm mẫu Grad-CAM.
- [x] Bổ sung test hardening; toàn bộ pytest và Ruff đã sạch ngày 02/10/2026.
- [ ] Huấn luyện E2-E4.
- [ ] Đánh giá Test E1-E4, tạo comparison và Grad-CAM nghiệm thu.
