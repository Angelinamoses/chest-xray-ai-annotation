
import numpy as np

from app.inference import CLASS_NAMES


TOP_K_FRACTION = 0.10


def calculate_binary_entropy(probabilities):
    """
    Calculate normalized binary entropy.

    Input:
        probabilities: NumPy array containing values [0, 1]

    Output:
        Entropy map containing values [0, 1]
    """

    epsilon = 1e-7

    p = np.clip(
        probabilities.astype(np.float32),
        epsilon,
        1.0 - epsilon
    )

    entropy = -(
        p * np.log2(p)
        + (1.0 - p) * np.log2(1.0 - p)
    )

    return entropy


def calculate_class_uncertainty(
    probability_map,
    top_k_fraction=TOP_K_FRACTION
):
    """
    Calculate uncertainty using the mean entropy
    of the most uncertain pixels.

    Returns a score between 0 and 1.
    """

    if not 0 < top_k_fraction <= 1:
        raise ValueError(
            "top_k_fraction must be between 0 and 1."
        )

    entropy_map = calculate_binary_entropy(
        probability_map
    )

    flattened_entropy = entropy_map.ravel()

    number_of_pixels = flattened_entropy.size

    k = max(
        1,
        int(np.ceil(number_of_pixels * top_k_fraction))
    )

    most_uncertain_pixels = np.partition(
        flattened_entropy,
        number_of_pixels - k
    )[-k:]

    return float(np.mean(most_uncertain_pixels))


def calculate_acquisition_score(probabilities):
    """
    Calculate image-level active learning score.

    Expected input shape:
        (5, 256, 256)

    Returns:
        image_score
        class_scores
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    expected_shape = (len(CLASS_NAMES), 256, 256)

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probabilities shape {expected_shape}, "
            f"received {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain NaN or infinite values."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    class_scores = {}

    for index, class_name in enumerate(CLASS_NAMES):

        score = calculate_class_uncertainty(
            probabilities[index]
        )

        class_scores[class_name] = score

    image_score = float(
        np.mean(list(class_scores.values()))
    )

    return {
        "acquisition_score": image_score,
        "class_uncertainty": class_scores,
        "scoring_method": "top_10_percent_binary_entropy"
    }
