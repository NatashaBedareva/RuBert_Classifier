from datasets import Dataset

from rubert_classifier.preprocessing import balance_dataset


def test_balance_dataset():
    ds = Dataset.from_dict(
        {
            "text": ["a"] * 5 + ["b"] * 20,
            "category": ["A"] * 5 + ["B"] * 20,
        }
    )
    balanced = balance_dataset(
        ds,
        label2id={"A": 0, "B": 1},
        list_of_categories=["A", "B"],
        target_samples_per_class=10,
        seed=0,
    )
    assert len(balanced) == 20
