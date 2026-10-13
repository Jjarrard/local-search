from localsearch.config import Config, config_path


def test_defaults_when_no_file_exists():
    cfg = Config.load()
    assert cfg.model is None
    assert cfg.base_url == "http://127.0.0.1:1338/v1"


def test_save_then_load_roundtrip():
    cfg = Config.load()
    cfg.model = "tiny"
    cfg.top = 5
    cfg.save()
    reloaded = Config.load()
    assert reloaded.model == "tiny"
    assert reloaded.top == 5


def test_corrupt_config_falls_back_to_defaults():
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert Config.load().model is None


def test_unknown_keys_are_ignored():
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"model": "m", "from_the_future": 1}')
    assert Config.load().model == "m"


def test_recent_models_order_first():
    cfg = Config()
    cfg.remember_model("b")
    cfg.remember_model("a")
    assert cfg.order_models(["c", "b", "a"]) == ["a", "b", "c"]
