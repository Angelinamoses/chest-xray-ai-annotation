
import numpy as np

from app.active_learning import (
    calculate_binary_entropy,
    calculate_class_uncertainty,
    calculate_acquisition_score,
)


def test_certain_predictions():
    """Predictions close to 0 or 1 should have low entropy."""

    probabilities = np.zeros((256, 256), dtype=np.float32)

    score = calculate_class_uncertainty(probabilities)

    assert score < 0.001, f"Unexpected score: {score}"

    print(f"PASS: Certain predictions | Score: {score:.6f}")


def test_uncertain_predictions():
    """Predictions of 0.5 should have maximum entropy."""

    probabilities = np.full(
        (256, 256),
        0.5,
        dtype=np.float32
    )

    score = calculate_class_uncertainty(probabilities)

    assert np.isclose(score, 1.0, atol=1e-5), (
        f"Expected approximately 1.0, got {score}"
    )

    print(f"PASS: Maximum uncertainty | Score: {score:.6f}")


def test_mixed_predictions():
    """A mixed probability map should produce an intermediate score."""

    probabilities = np.zeros((256, 256), dtype=np.float32)

    probabilities[:128, :] = 0.5

    score = calculate_class_uncertainty(probabilities)

    assert 0 < score <= 1

    print(f"PASS: Mixed predictions | Score: {score:.6f}")


def test_acquisition_score():
    """Test image-level scoring across all five classes."""

    probabilities = np.random.default_rng(42).uniform(
        0,
        1,
        size=(5, 256, 256)
    ).astype(np.float32)

    result = calculate_acquisition_score(probabilities)

    assert 0 <= result["acquisition_score"] <= 1

    assert len(result["class_uncertainty"]) == 5

    assert result["scoring_method"] == (
        "top_10_percent_binary_entropy"
    )

    print(
        "PASS: Image acquisition score | "
        f"Score: {result['acquisition_score']:.6f}"
    )


def test_invalid_shape():
    """Incorrect input dimensions must be rejected."""

    probabilities = np.zeros((3, 256, 256), dtype=np.float32)

    try:
        calculate_acquisition_score(probabilities)
    except ValueError:
        print("PASS: Invalid probability shape rejected")
    else:
        raise AssertionError("Invalid shape was not rejected")


def test_invalid_probabilities():
    """Probabilities outside [0,1] must be rejected."""

    probabilities = np.zeros((5, 256, 256), dtype=np.float32)

    probabilities[0, 0, 0] = 1.5

    try:
        calculate_acquisition_score(probabilities)
    except ValueError:
        print("PASS: Invalid probability values rejected")
    else:
        raise AssertionError("Invalid probability was not rejected")


if __name__ == "__main__":

    print("\nACTIVE LEARNING UNIT TESTS\n")

    test_certain_predictions()
    test_uncertain_predictions()
    test_mixed_predictions()
    test_acquisition_score()
    test_invalid_shape()
    test_invalid_probabilities()

    print("\n--------------------------------")
    print("ALL ACTIVE LEARNING TESTS PASSED!")
    print("--------------------------------")
