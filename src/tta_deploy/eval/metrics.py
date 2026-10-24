import numpy as np

from tta_deploy.data.imagenet_c import CORRUPTIONS
from tta_deploy.model import NUM_CLASSES

HEALTH_WINDOW = 500
HEALTH_EVERY = 50


def per_corruption_accuracy(
    predictions: np.ndarray, labels: np.ndarray, corruptions: np.ndarray
) -> dict[str, float]:
    assert predictions.shape == labels.shape == corruptions.shape
    accuracy = {
        c: float(
            (predictions[corruptions == c] == labels[corruptions == c]).mean() * 100
        )
        for c in CORRUPTIONS
    }
    accuracy["average"] = float(np.mean([accuracy[c] for c in CORRUPTIONS]))
    return accuracy


def rolling_windows(num_frames: int) -> list[slice]:
    return [
        slice(end - HEALTH_WINDOW, end)
        for end in range(HEALTH_WINDOW, num_frames + 1, HEALTH_EVERY)
    ]


def rolling_accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    correct = predictions == labels
    return np.array([correct[w].mean() * 100 for w in rolling_windows(len(correct))])


def rolling_prediction_entropy(entropies: np.ndarray) -> np.ndarray:
    return np.array([entropies[w].mean() for w in rolling_windows(len(entropies))])


def rolling_class_histogram_entropy(predictions: np.ndarray) -> np.ndarray:
    # Collapse indicator: falls towards 0 when predictions concentrate
    # on a few classes.
    values = []
    for w in rolling_windows(len(predictions)):
        p = np.bincount(predictions[w], minlength=NUM_CLASSES) / HEALTH_WINDOW
        p = p[p > 0]
        values.append(float(-(p * np.log(p)).sum()))
    return np.array(values)


def window_end_frames(num_frames: int) -> np.ndarray:
    return np.array([w.stop for w in rolling_windows(num_frames)])
