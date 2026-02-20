# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt

# print("-------------Preprocessing the data.-----------------------")
# Loading the dataset and splitting it into train, validation, and test sets
splits = {"train": "train.jsonl", "test": "test.jsonl"}
temp_train_set = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
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

print("-----------------------------------------------------------")
print("--------------------Logistic regression--------------------")
print("-----------------------------------------------------------")


def grid_search_lr(param_grid: dict, x_train, y_train: pd.DataFrame, x_val, y_val: pd.DataFrame, max_iter: int = 100):
    """
    Performs a grid search to find the best hyperparameters for the logistic regression model.

    Args:
        param_grid (dict): A dictionary containing the hyperparameters.
        x_train: The training features.
        y_train (pd.DataFrame): The training labels.
        x_val: The validation features.
        y_val (pd.DataFrame): The validation labels.
        max_iter (int): The maximum number of iterations for the model. Defaults to 100.

    Returns:
        best_model: The model with the best hyperparameters.
        best_macrof1: The best macro F1 score achieved with the best hyperparameters.
        best_params: The best hyperparameters found during the grid search.
    """
    # Initializing variables for the best model, best macro F1 score, and best hyperparameters
    best_model = None
    best_macrof1 = -1.0
    best_params = None

    # Iterating through each combinations of the hyperparameters
    for c in param_grid["C"]:
        for ratio in param_grid["l1_ratio"]:
            # Training the logistic regression model with the current hyperparameters
            model = LogisticRegression(
                C=c,
                solver="saga",
                l1_ratio=ratio,
                max_iter=max_iter,
                random_state=42,
            )
            
            # Fitting the model to the training data
            model.fit(x_train, y_train)

            # Predicting the labels for the validation set
            pred = model.predict(x_val)

            # Evaluating the model using the macro F1 score
            score = f1_score(y_val, pred, average="macro", zero_division=0)

            # Updating the relevant variables if the new model is better
            if score > best_macrof1:
                best_macrof1 = score
                best_model = model
                best_params = {"C": c, "l1_ratio": ratio}

    # Returning the best model, best macro F1 score, and best hyperparameters
    return best_model, best_macrof1, best_params

# Defining the possible hyperparameters for the grid search
param_grid = {
    "C": [0.1, 1],
    "l1_ratio": [0, 1],
}

# Finding the best model, f1 score, and hyperparameters 
best_model, best_macrof1, best_params = grid_search_lr(
    param_grid, X_train, y_train, X_val, y_val
)

# Printing the results of the search
print(best_model)
print(best_macrof1)
print(best_params)

# Defining the label names (classes)
label_names = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# Predicting the labels for the validation set using the found best model
y_pred_val_lr = best_model.predict(X_val)

# Classification report
print(
    classification_report(
        y_val, y_pred_val_lr, target_names=[label_names[i] for i in sorted(label_names)]
    )
)


# Confusion matrix
conf_matrix_val_lr = confusion_matrix(y_val, y_pred_val_lr)

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_val_lr, display_labels=[label_names[i] for i in sorted(label_names)]
)

# Plotting the confusion matrix
disp.plot(xticks_rotation="vertical")
plt.title("Validation Set Confusion Matrix: Logistic Regression")
plt.tight_layout()
plt.show()

# Predicting the labels for the test set using the found best model
y_pred_test_lr = best_model.predict(X_test)

# Classification report
print(
    classification_report(
        y_test, y_pred_test_lr, target_names=[label_names[i] for i in sorted(label_names)]
    )
)

# Confusion matrix
conf_matrix_test_lr = confusion_matrix(y_test, y_pred_test_lr)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_test_lr, display_labels=[label_names[i] for i in sorted(label_names)]
)

# Plotting the confusion matrix
disp.plot(xticks_rotation="vertical")
plt.title("Test set Confusion Matrix: Logistic Regression")
plt.tight_layout()
plt.show()

print("-----------------------------------------------------------")
print("---------------------------SVM-----------------------------")
print("-----------------------------------------------------------")

def grid_search_svm(param_grid: dict, x_train, y_train: pd.DataFrame, x_val, y_val: pd.DataFrame, max_iter: int=100):
    """
    Performs a grid search to find the best hyperparameters for the linear SVM model.

    Args:
        param_grid (dict): A dictionary containing the hyperparameters.
        x_train: The training features.
        y_train (pd.DataFrame): The training labels.
        x_val: The validation features.
        y_val (pd.DataFrame): The validation labels.
        max_iter (int): The maximum number of iterations for the model. Defaults to 100.

    Returns:
        best_model: The model with the best hyperparameters.
        best_macrof1: The best macro F1 score achieved with the best hyperparameters.
        best_params: The best hyperparameters found during the grid search.
    """
    # Initializing variables for the best model, best macro F1 score, and best hyperparameters
    best_model = None
    best_macrof1 = -1.0
    best_params = None

    # Iterating through each combinations of the hyperparameters
    for c in param_grid["C"]:
        for loss in param_grid["loss"]:
            # Training the logistic regression model with the current hyperparameters
            model = LinearSVC(
                C=c,
                loss=loss,
                max_iter=max_iter,
                random_state=42,
            )

            # Fitting the model to the training data
            model.fit(x_train, y_train)

            # Predicting the labels for the validation set
            pred = model.predict(x_val)

            # Evaluating the model using the macro F1 score
            score = f1_score(y_val, pred, average="macro", zero_division=0)

            # Updating the relevant variables if the new model is better
            if score > best_macrof1:
                best_macrof1 = score
                best_model = model
                best_params = {
                    "C": c,
                    "loss": loss,
                }

    # Returning the best model, best macro F1 score, and best hyperparameters
    return best_model, best_macrof1, best_params

# Defining the possible hyperparameters for the grid search
param_grid_svm = {"C": [0.1, 1], "loss": ["hinge", "squared_hinge"]}

# Finding the best model, f1 score, and hyperparameters
best_model_svm, best_macrof1_svm, best_params_svm = grid_search_svm(
    param_grid_svm, X_train, y_train, X_val, y_val
)

# Printing the results of the search
print(best_model_svm)
print(best_macrof1_svm)
print(best_params_svm)

# Predicting the labels for the validation set using the found best model
y_pred_val_svm = best_model_svm.predict(X_val)

# Classification report
print(
    classification_report(
        y_val,
        y_pred_val_svm,
        target_names=[label_names[i] for i in sorted(label_names)],
    )
)

# Confusion matrix
conf_matrix_val_svm = confusion_matrix(y_val, y_pred_val_svm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_val_svm, display_labels=[label_names[i] for i in sorted(label_names)]
)

# Plotting the confusion matrix
disp.plot(xticks_rotation="vertical")
plt.title("Validation set Confusion Matrix: Linear SVM")
plt.tight_layout()
plt.show()

# Predicting the labels for the test set using the found best model
y_pred_test_svm = best_model_svm.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred_test_svm,
        target_names=[label_names[i] for i in sorted(label_names)],
    )
)

# Confusion matrix
conf_matrix_test_svm = confusion_matrix(y_test, y_pred_test_svm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_test_svm, display_labels=[label_names[i] for i in sorted(label_names)]
)

# Plotting the confusion matrix
disp.plot(xticks_rotation="vertical")
plt.title("Test set Confusion Matrix: Linear SVM")
plt.tight_layout()
plt.show()

print("-----------------------------------------------------------")
print("----------------------Error analysis-----------------------")
print("-----------------------------------------------------------")

pd.set_option("display.max_colwidth", None)

# Error Analysis for Logistic Regression
df_predictions_lr = pd.DataFrame(
    {
        "title": ag_test["title"],
        "description": ag_test["description"],
        "true_label": y_test.map(label_names),
        "pred_label": pd.Series(y_pred_test_lr).map(label_names).values,
    }
)

# Identifying the misclassified Logistic Regression samples
errors_lr = df_predictions_lr[
    df_predictions_lr["true_label"] != df_predictions_lr["pred_label"]
]

# Displaying the first 20 misclassified samples
print("Logistic Regression")
print(f"Total Errors: {len(errors_lr)}")
display(errors_lr.head(20))

# Error Analysis for Linear SVM
df_predictions_svm = pd.DataFrame(
    {
        "title": ag_test["title"],
        "description": ag_test["description"],
        "true_label": y_test.map(label_names),
        "pred_label": pd.Series(y_pred_test_svm).map(label_names).values,
    }
)

# Identifying the misclassified Linear SVM samples
errors_svm = df_predictions_svm[
    df_predictions_svm["true_label"] != df_predictions_svm["pred_label"]
]

# Displaying the first 20 misclassified samples
print("\nLinear SVM")
print(f"Total Errors: {len(errors_svm)}")
display(errors_svm.head(20))

