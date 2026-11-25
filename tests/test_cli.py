from spamdetector.cli import build_parser, main


def test_parser_knows_commands():
    parser = build_parser()
    assert parser.parse_args(["train", "--quick"]).quick is True
    assert parser.parse_args(["predict", "hello"]).messages == ["hello"]


def test_predict_without_model_returns_error_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("spamdetector.predict.MODEL_PATH", tmp_path / "none.joblib")
    assert main(["predict", "hello"]) == 1
    assert "spam-detector train" in capsys.readouterr().err
