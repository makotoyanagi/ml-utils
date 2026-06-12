import numpy as np


def accuracy(y_true, y_pred):
    """Classification accuracy."""
    return np.mean(np.array(y_true) == np.array(y_pred))


def precision(y_true, y_pred, average="macro"):
    """Precision score (macro or per-class)."""
    classes = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
    if average == "macro":
        return np.mean(precisions)
    return dict(zip(classes, precisions))


def recall(y_true, y_pred, average="macro"):
    """Recall score (macro or per-class)."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
    if average == "macro":
        return np.mean(recalls)
    return dict(zip(classes, recalls))


def f1_score(y_true, y_pred, average="macro"):
    """F1 score."""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    if isinstance(p, dict):
        return {c: 2*p[c]*r[c]/(p[c]+r[c]) if (p[c]+r[c]) > 0 else 0.0 for c in p}
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0


def mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def mae(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
