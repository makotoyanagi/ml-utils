# ML Utils

Lightweight ML utility library for preprocessing, metrics, and visualization.

## Features
- Data Preprocessing: normalization, encoding, splitting
- Metrics: accuracy, precision, recall, F1
- Visualization: training curves, confusion matrices

## Quick Start
```python
from ml_utils.preprocessing import normalize, train_test_split
from ml_utils.metrics import accuracy, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## License
MIT
