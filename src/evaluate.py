"""Model evaluation utilities: metrics computation and confusion matrix plotting."""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support
)


def compute_metrics(y_test, y_pred):
    """Compute accuracy, macro and weighted precision/recall/F1.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Metric name -> value mapping.
    """
    acc = accuracy_score(y_test, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    return {
        'accuracy': acc,
        'precision_macro': prec_macro, 'recall_macro': rec_macro, 'f1_macro': f1_macro,
        'precision_weighted': prec_w, 'recall_weighted': rec_w, 'f1_weighted': f1_w
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Train a model and evaluate it on the test set, recording timing.

    Args:
        model: Unfitted sklearn estimator.
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        dict: Contains fitted model, predictions, metrics, and timing info.
    """
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer

    metrics = compute_metrics(y_test, y_pred)

    return {
        'model': model, 'y_pred': y_pred, 'metrics': metrics,
        'train_time': train_time, 'infer_time': infer_time
    }


def plot_confusion_matrix(y_test, y_pred, labels, title, save_path=None):
    """Plot a confusion matrix as a heatmap.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): Ordered list of class labels.
        title (str): Plot title.
        save_path (str, optional): If given, saves the figure to this path.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)
