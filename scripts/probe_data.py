import numpy as np
from torch.utils.data import Dataset
import logging
import json
from sklearn.utils import resample
from functools import partial


def balance_dataset(dataset, seed):
    """
    Balance the dataset by undersampling the majority class.
    """
    labels = np.array([item["label"] for item in dataset])
    # breakpoint()
    positive_samples = [item for item in dataset if item["label"] == 1]
    negative_samples = [item for item in dataset if item["label"] == 0]

    logging.info(f"Before balancing:")
    logging.info(f"  Total samples: {len(dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")

    if not positive_samples or not negative_samples:
        logging.warning("No positive samples found.")
        return None

    if len(positive_samples) < len(negative_samples):
        negative_samples = resample(
            negative_samples,
            n_samples=len(positive_samples),
            random_state=seed,
        )
    else:
        positive_samples = resample(
            positive_samples,
            n_samples=len(negative_samples),
            random_state=seed,
        )

    balanced_dataset = positive_samples + negative_samples
    np.random.shuffle(balanced_dataset)

    logging.info(f"After balancing:")
    logging.info(f"  Total samples: {len(balanced_dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")

    return balanced_dataset


class ProbingDataset(Dataset):
    def __init__(self, jsonl_file, filter_criterion):
        self.sentences = []
        self.labels = []
        self.filter_criterion = filter_criterion
        self.load_data(jsonl_file)

    def load_data(self, jsonl_file):
        with open(jsonl_file, "r") as in_data:
            for line in in_data:
                obj = json.loads(line)
                label = 1 if self.filter_criterion(obj) else 0
                self.sentences.append(obj["sentence"])
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "label": self.labels[idx]}


class MulticlassProbingDataset(Dataset):
    def __init__(self, jsonl_file, concept):
        self.sentences = []
        self.labels = []
        self.concept = concept
        self.classes = {
            "domain": ["fantasy", "science", "news", "other"],
            "sentiment": ["positive", "neutral", "negative"],
            "voice": ["active", "passive"],
            "tense": ["present", "past"],
            "formality": ["high", "neutral"],
        }
        self.load_data(jsonl_file)

    def load_data(self, jsonl_file):
        with open(jsonl_file, "r") as in_data:
            for line in in_data:
                obj = json.loads(line)
                label = self.classes[self.concept].index(obj[self.concept])
                self.sentences.append(obj["sentence"])
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "label": self.labels[idx]}


def concept_filter(example_dict, concept_key, concept_value):
    for key in example_dict.keys():
        if concept_key == key and example_dict[key] == concept_value:
            return True
    return False


def concept_filter_multiclass(example_dict, concept_key, concept_value):
    classes = {
        "domain": ["fantasy", "science", "news", "other"],
        "sentiment": ["positive", "neutral", "negative"],
        "voice": ["active", "passive"],
        "tense": ["present", "past"],
        "formality": ["high", "neutral"],
    }
    return classes[concept_key].index(concept_value)


def prepare_datasets(
    train_filepath, test_filepath, concept_key, concept_value, label_type, seed
):
    """Prepare and balance the training and test datasets."""
    if label_type == "binary":
        filter_func = concept_filter
    elif label_type == "multiclass":
        filter_func = concept_filter_multiclass
    filter_criterion = partial(
        filter_func, concept_key=concept_key, concept_value=concept_value
    )
    train_dataset = ProbingDataset(train_filepath, filter_criterion)
    test_dataset = ProbingDataset(test_filepath, filter_criterion)

    print("Balancing training dataset...")
    train_dataset = balance_dataset(train_dataset, seed)
    print("Balancing test dataset...")
    test_dataset = balance_dataset(test_dataset, seed)

    return train_dataset, test_dataset
