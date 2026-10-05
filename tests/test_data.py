from rubert_classifier.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.dataset_name == "Davlan/sib200"
    assert cfg.seed == 42


def test_config_save_load(tmp_path):
    cfg = Config()
    cfg.list_of_categories = ["a", "b"]
    cfg.num_labels = 2
    path = tmp_path / "cfg.json"
    cfg.save(str(path))

    loaded = Config.load(str(path))
    assert loaded.list_of_categories == ["a", "b"]
    assert loaded.num_labels == 2