import numpy as np
from torch.utils.data import Dataset
import logging
import json
from sklearn.utils import resample
from functools import partial

import json
import numpy as np
from torch.utils.data import Dataset
import logging


class PairwiseProbingDataset(Dataset):
    """
    Pairwise dataset for direction-invariant *change* of a binary concept.

    Each item:
        {
            "sentence_1": str,
            "sentence_2": str,
            "label": int  # 1 if concept differs between s1 and s2, else 0
        }

    Concept is defined by (concept_key == concept_value).
    """

    def __init__(self, jsonl_file, concept_key, concept_value, seed=0):
        self.sentences = []
        self.single_labels = []  # 0/1 per sentence
        self.pairs = []  # list of (idx1, idx2, label)
        self.concept_key = concept_key
        self.concept_value = concept_value
        self.seed = seed

        self._load_single_sentences(jsonl_file)
        self._build_pairs()

    def _load_single_sentences(self, jsonl_file):
        """Load sentences and compute 0/1 label per sentence for the given concept."""
        with open(jsonl_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                sent = obj["sentence"]
                label = (
                    1
                    if (
                        self.concept_key in obj
                        and obj[self.concept_key] == self.concept_value
                    )
                    else 0
                )
                self.sentences.append(sent)
                self.single_labels.append(label)

        self.single_labels = np.array(self.single_labels, dtype=int)

    def _build_pairs(self):
        """Build balanced change vs no-change pairs."""
        rng = np.random.default_rng(self.seed)

        pos_indices = np.where(self.single_labels == 1)[0].tolist()
        neg_indices = np.where(self.single_labels == 0)[0].tolist()

        logging.info(
            f"Pairwise dataset (binary): {len(self.single_labels)} sentences "
            f"({len(pos_indices)} positive, {len(neg_indices)} negative) "
            f"for concept {self.concept_key}={self.concept_value}"
        )

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            logging.warning(
                "Not enough positives or negatives to build pairwise dataset."
            )
            self.pairs = []
            return

        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        n = min(len(pos_indices), len(neg_indices))
        pos_indices = pos_indices[:n]
        neg_indices = neg_indices[:n]

        # --- CHANGE pairs: y1!=y2, label = 1 ---
        change_pairs = []
        for i_pos, i_neg in zip(pos_indices, neg_indices):
            change_pairs.append((i_pos, i_neg, 1))
            change_pairs.append((i_neg, i_pos, 1))  # both directions

        # --- NO-CHANGE pairs: (pos,pos) & (neg,neg), label = 0 ---
        no_change_pairs = []

        # Sample same-class pairs (without pairing an index with itself)
        def sample_same_pairs(indices, target_count):
            pairs = []
            if len(indices) < 2:
                return pairs
            for _ in range(target_count):
                i1, i2 = rng.choice(indices, size=2, replace=False)
                pairs.append((i1, i2, 0))
            return pairs

        # We want as many no-change pairs as change pairs
        target_no_change = len(change_pairs)
        half = target_no_change // 2
        no_change_pairs += sample_same_pairs(pos_indices, half)
        no_change_pairs += sample_same_pairs(
            neg_indices, target_no_change - len(no_change_pairs)
        )

        # Truncate if we somehow overshot
        no_change_pairs = no_change_pairs[:target_no_change]

        self.pairs = change_pairs + no_change_pairs
        rng.shuffle(self.pairs)

        logging.info(
            f"Built {len(self.pairs)} pairs ("
            f"{len(change_pairs)} change, {len(no_change_pairs)} no-change) "
            f"for concept {self.concept_key}={self.concept_value}"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        return {
            "sentence_1": self.sentences[idx1],
            "sentence_2": self.sentences[idx2],
            "label": int(label),
        }


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
    CLASSES = {
        "domain": ["fantasy", "science", "news", "other"],
        "sentiment": ["positive", "neutral", "negative"],
        "voice": ["active", "passive"],
        "tense": ["present", "past"],
        "formality": ["high", "neutral"],
    }

    def __init__(self, jsonl_file, concept):
        if concept not in self.CLASSES:
            raise ValueError(
                f"Unknown concept '{concept}'. Valid: {list(self.CLASSES.keys())}"
            )

        self.sentences = []
        self.labels = []
        self.concept = concept
        self.classes = self.CLASSES[concept]
        self.load_data(jsonl_file)

    def load_data(self, jsonl_file):
        with open(jsonl_file, "r") as in_data:
            for line in in_data:
                obj = json.loads(line)
                if self.concept not in obj:
                    # skip examples without this concept annotation
                    continue
                value = obj[self.concept]
                if value not in self.classes:
                    # optionally warn, but skip unknown values
                    logging.warning(
                        f"Unknown value '{value}' for concept '{self.concept}', skipping example."
                    )
                    continue
                label = self.classes.index(value)
                self.sentences.append(obj["sentence"])
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "label": self.labels[idx]}


class PairwiseMulticlassProbingDataset(Dataset):
    """
    Pairwise dataset for direction-invariant *change* in a multiclass concept.

    Each item:
        {
            "sentence_1": str,
            "sentence_2": str,
            "label": int  # 1 if class differs between s1 and s2, else 0
        }

    concept ∈ {"domain", "sentiment", "voice", "tense", "formality"}.
    """

    CLASSES = MulticlassProbingDataset.CLASSES

    def __init__(self, jsonl_file, concept, seed=0):
        if concept not in self.CLASSES:
            raise ValueError(
                f"Unknown concept '{concept}'. Valid: {list(self.CLASSES.keys())}"
            )

        self.sentences = []
        self.labels = []  # class index per sentence
        self.concept = concept
        self.classes = self.CLASSES[concept]
        self.seed = seed
        self.pairs = []

        self._load_single_sentences(jsonl_file)
        self._build_pairs()

    def _load_single_sentences(self, jsonl_file):
        with open(jsonl_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                if self.concept not in obj:
                    continue
                value = obj[self.concept]
                if value not in self.classes:
                    logging.warning(
                        f"Unknown value '{value}' for concept '{self.concept}', skipping."
                    )
                    continue
                label = self.classes.index(value)
                self.sentences.append(obj["sentence"])
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=int)

    def _build_pairs(self):
        rng = np.random.default_rng(self.seed)

        n = len(self.labels)
        if n < 2:
            logging.warning(
                "Not enough examples to build pairwise multiclass dataset."
            )
            return

        # Group indices by class
        idx_by_class = {
            c: np.where(self.labels == c)[0].tolist()
            for c in range(len(self.classes))
        }

        logging.info(
            f"Pairwise dataset (multiclass): {n} sentences across "
            f"{len(self.classes)} classes for concept '{self.concept}'."
        )

        # Build CHANGE pairs (y1 != y2, label=1)
        change_pairs = []
        # simplest: sample across different classes, limited by smallest class size
        # build a pool of indices
        all_indices = np.arange(n)

        # target number of change pairs: heuristically 2 * n
        target_change = 2 * n
        attempts = 0
        max_attempts = 10 * target_change

        while len(change_pairs) < target_change and attempts < max_attempts:
            i1, i2 = rng.choice(all_indices, size=2, replace=False)
            if self.labels[i1] != self.labels[i2]:
                change_pairs.append((i1, i2, 1))
            attempts += 1

        # Build NO-CHANGE pairs (same class, distinct examples), label=0
        no_change_pairs = []
        target_no_change = len(change_pairs)
        for c, indices in idx_by_class.items():
            if len(indices) < 2:
                continue
            # sample same-class pairs
            num_pairs_c = min(
                target_no_change - len(no_change_pairs), len(indices)
            )
            for _ in range(num_pairs_c):
                i1, i2 = rng.choice(indices, size=2, replace=False)
                no_change_pairs.append((i1, i2, 0))
                if len(no_change_pairs) >= target_no_change:
                    break
            if len(no_change_pairs) >= target_no_change:
                break

        # If we didn't get enough no-change pairs, just truncate change_pairs too
        m = min(len(change_pairs), len(no_change_pairs))
        change_pairs = change_pairs[:m]
        no_change_pairs = no_change_pairs[:m]

        self.pairs = change_pairs + no_change_pairs
        rng.shuffle(self.pairs)

        logging.info(
            f"Built {len(self.pairs)} multiclass pairs "
            f"({len(change_pairs)} change, {len(no_change_pairs)} no-change) "
            f"for concept '{self.concept}'."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        return {
            "sentence_1": self.sentences[idx1],
            "sentence_2": self.sentences[idx2],
            "label": int(label),
        }


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


def prepare_pairwise_datasets(
    train_filepath,
    test_filepath,
    concept_key,
    label_type,
    concept_value=None,
    seed=0,
):
    """
    Prepare pairwise datasets for SSAE-style Δ-probing.

    label_type:
        - "binary": concept_key + concept_value define a binary concept.
        - "multiclass": concept_key is e.g. "domain"/"sentiment"/..., concept_value unused.

    Returns:
        train_dataset, test_dataset
    """
    if label_type == "binary":
        if concept_value is None:
            raise ValueError(
                "concept_value must be provided for binary pairwise datasets."
            )
        train_dataset = PairwiseProbingDataset(
            train_filepath,
            concept_key=concept_key,
            concept_value=concept_value,
            seed=seed,
        )
        test_dataset = PairwiseProbingDataset(
            test_filepath,
            concept_key=concept_key,
            concept_value=concept_value,
            seed=seed + 1,
        )
    elif label_type == "multiclass":
        train_dataset = PairwiseMulticlassProbingDataset(
            train_filepath,
            concept=concept_key,
            seed=seed,
        )
        test_dataset = PairwiseMulticlassProbingDataset(
            test_filepath,
            concept=concept_key,
            seed=seed + 1,
        )
    else:
        raise ValueError(
            f"Unknown label_type '{label_type}', must be 'binary' or 'multiclass'."
        )

    logging.info(
        f"Pairwise train size: {len(train_dataset)}, "
        f"test size: {len(test_dataset)} "
        f"for concept_key='{concept_key}', label_type='{label_type}'."
    )

    return train_dataset, test_dataset


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
