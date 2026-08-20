from babeldoc.docvision import doclayout


def test_model_loading_callback_runs_after_asset_resolution_before_model_init(
    monkeypatch, tmp_path
):
    events = []
    model_path = tmp_path / "model.onnx"

    def record_model_init(_model, _path):
        events.append("model_init")

    monkeypatch.setattr(
        doclayout,
        "get_doclayout_onnx_model_path",
        lambda: events.append("assets_ready") or model_path,
    )
    monkeypatch.setattr(
        doclayout.OnnxModel,
        "__init__",
        record_model_init,
    )

    doclayout.OnnxModel.from_pretrained(lambda: events.append("loading_model"))

    assert events == ["assets_ready", "loading_model", "model_init"]
