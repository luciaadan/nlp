# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the dataset and splitting it into train, validation, and test sets
splits = {"train": "train.jsonl", "test": "test.jsonl"}
temp_train_set = pd.read_json(
    "hf://datasets/sh0416/ag_news/" + splits["train"], lines=True
)
test_set = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)

# Splitting the train set into train and validation set with 9:1 ratio.
train_set, val_set = train_test_split(
    temp_train_set,
    test_size=0.1,
    random_state=42,
    stratify=temp_train_set["label"],
)


def separate_labels_text(set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates the labels and text from the given DataFrame.

    Args:
        set (pd.DataFrame): The input DataFrame containing the text and labels.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the features and the labels.
    """
    x = set.drop(columns=["label"])
    y = set["label"]

    return x, y


# Separating the labels and text for train, validation, and test sets
ag_train, y_train = separate_labels_text(train_set)
ag_val, y_val = separate_labels_text(val_set)
ag_test, y_test = separate_labels_text(test_set)

# Vectorizing the text data using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

# Transforming the text data into TF-IDF features
X_train = tfidf.fit_transform(ag_train["title"] + " " + ag_train["description"])
X_val = tfidf.transform(ag_val["title"] + " " + ag_val["description"])
X_test = tfidf.transform(ag_test["title"] + " " + ag_test["description"])
