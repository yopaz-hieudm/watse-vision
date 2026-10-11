from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    (
        "app/streamlit_app.py",
        "app/pages/0_Tong_quan.py",
        "app/pages/1_Du_lieu.py",
        "app/pages/2_Huan_luyen.py",
        "app/pages/3_Danh_gia.py",
        "app/pages/4_So_sanh.py",
        "app/pages/5_Du_doan.py",
    ),
)
def test_streamlit_page_renders_without_exception(relative_path: str) -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app/streamlit_app.py", default_timeout=30)
    app.run()
    if relative_path != "app/streamlit_app.py":
        app.switch_page(relative_path.removeprefix("app/")).run()
    assert not app.exception


def test_prediction_history_can_be_reviewed_even_without_current_checkpoint(tmp_path, monkeypatch):
    import io

    from PIL import Image

    import realwaste.inference.history as history_module

    history = history_module.PredictionHistory(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"old-model")
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), "red").save(buffer, "PNG")
    identifier = history.save(
        experiment_id="E1",
        checkpoint=checkpoint,
        filename="review.png",
        image=buffer.getvalue(),
        predictions=[{"class_name": "Glass", "probability": 0.5}],
        elapsed_ms=10,
        retain_image=True,
    )
    monkeypatch.setattr(history_module, "PredictionHistory", lambda _root: history)
    app = AppTest.from_file(PROJECT_ROOT / "app/streamlit_app.py", default_timeout=30).run()
    app.switch_page("pages/5_Du_doan.py").run()
    assert not app.exception
    label_selector = next(item for item in app.selectbox if item.label == "Nhãn đúng")
    label_selector.select("Paper")
    next(
        item for item in app.checkbox if item.label == "Tôi đã kiểm tra ảnh và xác nhận nhãn này"
    ).check()
    next(item for item in app.button if item.label == "Lưu nhãn xác nhận").click().run()
    assert not app.exception
    assert history.list("E1")[0]["reviewed_label"] == "Paper"
    next(item for item in app.button if item.label == "Bỏ xác nhận").click().run()
    assert not app.exception
    assert history.get_image(identifier) is not None
    assert history.reviewed_count() == 0
