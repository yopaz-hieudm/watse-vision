# RealWaste classifier

This project implements reproducible RealWaste image-classification experiments with
MobileNetV3 Small and ResNet18. It covers deterministic data splits, transfer learning,
augmentation comparisons, Test metrics, confusion matrices, error analysis and Grad-CAM.

## Requirements

- Python 3.11 or 3.12
- Optional NVIDIA GPU for faster training
- About 1 GB for the dataset archive and extracted images, plus model outputs

## Install

Using `uv`:

```bash
uv sync --extra dev
```

Activate the environment or prefix commands with `uv run`.

To install the web interface as well:

```bash
uv sync --extra dev --extra app
```

## Data

The official source is UCI Machine Learning Repository Dataset ID 908.

For the first trusted download:

```bash
uv run python scripts/download_data.py --record-checksum
```

Review and commit `data/checksums/realwaste.sha256`. Later runs verify that hash:

```bash
uv run python scripts/download_data.py --verify-only
```

Generate EDA artifacts and deterministic splits:

```bash
uv run python scripts/inspect_data.py --data-dir data/raw/RealWaste --output-dir outputs/eda
uv run python scripts/create_splits.py --data-dir data/raw/RealWaste --seed 42
```

## Mandatory experiments

```bash
uv run python scripts/train.py --config configs/mobilenet_v3_small_baseline.yaml
uv run python scripts/train.py --config configs/resnet18_baseline.yaml
uv run python scripts/train.py --config configs/mobilenet_v3_small_aug.yaml
uv run python scripts/train.py --config configs/resnet18_aug.yaml
```

A fresh run is rejected when the experiment directory already contains training
artifacts. Continue safely with `--resume outputs/experiments/<name>/last.pt`.
Use `--overwrite` only when you intentionally want to remove that experiment's
checkpoints, history, evaluation and Grad-CAM artifacts before retraining.

Evaluate each `best.pt` checkpoint on Test. Example:

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/experiments/mobilenet_v3_small_baseline/best.pt \
  --split test
```

After E1-E4 have `metrics.json` files:

```bash
uv run python scripts/compare_experiments.py
```

## Grad-CAM and inference

```bash
uv run python scripts/generate_gradcam.py \
  --checkpoint outputs/experiments/mobilenet_v3_small_aug/best.pt \
  --samples 20

uv run python scripts/predict.py \
  --checkpoint outputs/experiments/mobilenet_v3_small_aug/best.pt \
  --image sample.jpg \
  --top-k 3
```

## Web interface

Start the Vietnamese multi-page control center with:

```bash
uv run streamlit run app/streamlit_app.py
```

The interface provides dataset/EDA inspection, background E1-E4 training with live
progress, Validation/Test evaluation, experiment comparison and image prediction.
Training continues in a separate process, so Streamlit remains responsive. Only the
fixed E1-E4 configurations and project-local artifact paths are accepted.

On Windows, an existing environment can also launch the UI directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The workflow links and the suggested next step connect every page. The selected
experiment persists across pages; Prediction only offers compatible checkpoints
from that experiment. Finishing a background job refreshes page state automatically.

To recover an incompatible checkpoint or restart an experiment, open **Huấn luyện →
Mở phiên huấn luyện mới**. The entire previous run moves to
`outputs/archive/<E1-E4>/<timestamp>_<id>` before fresh training is enabled.
Archived runs are preserved; active jobs block this operation.

Prediction results persist in `outputs/predictions/history.sqlite3`, identified by
the image content and checkpoint SHA-256. Image retention is opt-in. In **Dự đoán →
Lịch sử và phản hồi**, review retained images and confirm or correct their labels.
To use those images, start a new training run and select **Bổ sung ảnh đã xác nhận**.
The job freezes a manifest and images in `outputs/feedback/snapshots/`; resume uses
that same immutable snapshot. Only the training loader gets extra images.
Validation/Test images and conflicting labels are rejected. Feedback images are
shared across experiments, while prediction history is filtered by experiment.
Comparison requires all four experiments to use the same feedback snapshot.

## Lifecycle verification

After downloading the verified dataset and creating the standard splits, run:

```bash
uv run python scripts/verify_lifecycle.py
```

This runs full dataset EDA, then E1-E4 training, Validation/Test evaluation,
comparison, prediction, persisted history, reviewed feedback, Grad-CAM and training
with a feedback snapshot. Training uses 36 real images, 18 Validation images,
18 Test images, 64px inputs and one epoch without pretrained weights. This verifies
the software lifecycle; it does not establish classifier accuracy.
The script writes isolated checkpoints, reports and logs in
`outputs/verification/<run-id>/`, publishes `outputs/verification/latest.json`,
and displays that report on **Tổng quan**. Standard E1-E4 checkpoints are separate.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

See `PLAN_CODE_REALWASTE.md` for the complete implementation and acceptance plan.
# watse-vision
