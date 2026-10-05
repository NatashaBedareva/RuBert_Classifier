from rubert_classifier.config import Config


def test_config_num_labels_field():
    cfg = Config()
    cfg.num_labels = 7
    cfg.id2label = {i: f"c{i}" for i in range(7)}
    assert cfg.num_labels == 7
    assert cfg.id2label[0] == "c0"