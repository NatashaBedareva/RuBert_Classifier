"""Preprocessing: balancing and tokenization."""
import random
from collections import Counter
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from rubert_classifier.config import Config


def balance_dataset(
    dataset: Dataset,
    label2id: dict,
    list_of_categories: list,
    target_samples_per_class: int = 150,
    seed: int = 42,
) -> Dataset:
    """Balance the dataset with oversampling of minority classes."""
    random.seed(seed)
    texts = dataset["text"]
    categories = dataset["category"]

    print("\nOriginal class distribution:")
    for category in list_of_categories:
        count = sum(1 for cat in categories if cat == category)
        print(f"  {category}: {count}")

    balanced_texts = []
    balanced_categories = []

    for category in list_of_categories:
        class_indices = [i for i, cat in enumerate(categories) if cat == category]
        class_texts = [texts[i] for i in class_indices]
        class_categories = [categories[i] for i in class_indices]

        if len(class_texts) < target_samples_per_class:
            repeat_times = target_samples_per_class // len(class_texts)
            remainder = target_samples_per_class % len(class_texts)

            balanced_texts.extend(class_texts * repeat_times)
            balanced_categories.extend(class_categories * repeat_times)

            if remainder > 0:
                indices = random.sample(range(len(class_texts)), remainder)
                balanced_texts.extend([class_texts[i] for i in indices])
                balanced_categories.extend([class_categories[i] for i in indices])
        else:
            indices = random.sample(range(len(class_texts)), target_samples_per_class)
            balanced_texts.extend([class_texts[i] for i in indices])
            balanced_categories.extend([class_categories[i] for i in indices])

    combined = list(zip(balanced_texts, balanced_categories))
    random.shuffle(combined)
    balanced_texts, balanced_categories = zip(*combined)

    print(f"\nAfter balancing: {len(balanced_texts)} examples")
    balanced_counts = Counter(balanced_categories)
    for category in list_of_categories:
        print(f"  {category}: {balanced_counts[category]}")

    return Dataset.from_dict(
        {"text": list(balanced_texts), "category": list(balanced_categories)}
    )


def build_tokenizer(cfg: Config) -> PreTrainedTokenizer:
    """Load the tokenizer."""
    return AutoTokenizer.from_pretrained(cfg.model_name)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    cfg: Config,
    add_labels: bool = True,
) -> Dataset:
    """Tokenize a dataset."""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=cfg.max_length,
            add_special_tokens=True,
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    if add_labels and cfg.label2id:
        tokenized = tokenized.add_column(
            "labels", [cfg.label2id[cat] for cat in tokenized["category"]]
        )

    return tokenized


def prepare_datasets(
    raw: DatasetDict, cfg: Config, tokenizer: Optional[PreTrainedTokenizer] = None
) -> Tuple[Dataset, Dataset, Dataset, PreTrainedTokenizer]:
    """Full pipeline: balance -> tokenize."""
    if tokenizer is None:
        tokenizer = build_tokenizer(cfg)

    balanced_train = balance_dataset(
        raw["train"],
        cfg.label2id,
        cfg.list_of_categories,
        target_samples_per_class=cfg.target_samples_per_class,
        seed=cfg.seed,
    )

    tokenized_train = tokenize_dataset(balanced_train, tokenizer, cfg)
    tokenized_val = tokenize_dataset(raw["validation"], tokenizer, cfg)
    tokenized_test = tokenize_dataset(raw["test"], tokenizer, cfg)

    print(
        f"\nTokenized sizes -> train: {len(tokenized_train)}, "
        f"val: {len(tokenized_val)}, test: {len(tokenized_test)}"
    )

    return tokenized_train, tokenized_val, tokenized_test, tokenizer