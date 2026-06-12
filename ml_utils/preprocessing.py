import numpy as np


def normalize(X, method="standard"):
    """Normalize features.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        method: 'standard' (z-score) or 'minmax' (0-1 scaling)
    
    Returns:
        Normalized array and fitted parameters
    """
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        return (X - mean) / std, {"mean": mean, "std": std}
    elif method == "minmax":
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0) + 1e-8
        return (X - xmin) / (xmax - xmin), {"min": xmin, "max": xmax}
    else:
        raise ValueError(f"Unknown method: {method}")


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays into random train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot_encode(y, num_classes=None):
    """One-hot encode integer labels."""
    if num_classes is None:
        num_classes = len(np.unique(y))
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot
