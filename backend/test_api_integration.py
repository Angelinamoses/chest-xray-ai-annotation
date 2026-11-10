
import base64
import io
import requests

from PIL import Image
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"

DICOM_PATH = (
    "../data/images/train/vindr_300/"
    "020717d6ab0b440b37978d9bace9f9b2.dicom"
)

CLASSES = [
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
    "Lung Opacity",
]


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"PASS: {message}")


def decode_png(encoded_string):

    image_bytes = base64.b64decode(
        encoded_string,
        validate=True
    )

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image.load()

    return np.asarray(image)


# --------------------------------------------------
# 1. HEALTH CHECK
# --------------------------------------------------

health = requests.get(
    f"{API_URL}/health",
    timeout=15
)

check(health.status_code == 200, "Health endpoint")

check(
    health.json()["model_loaded"] is True,
    "Model is loaded"
)


# --------------------------------------------------
# 2. PREDICTION REQUEST
# --------------------------------------------------

with open(DICOM_PATH, "rb") as dicom_file:

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "integration_test.dicom",
                dicom_file,
                "application/dicom"
            )
        },
        timeout=120
    )

check(
    response.status_code == 200,
    "Prediction endpoint returns HTTP 200"
)

result = response.json()


# --------------------------------------------------
# 3. RESPONSE STRUCTURE
# --------------------------------------------------

check(
    result["model_version"] == "unet-v1",
    "Model version"
)

for field in [
    "masks",
    "probabilities",
    "confidence",
    "uncertainty",
]:

    check(
        field in result,
        f"{field} field exists"
    )

for field in [
    "masks",
    "probabilities",
    "confidence",
    "uncertainty",
]:

    check(
        set(result[field].keys()) == set(CLASSES),
        f"All five classes exist in {field}"
    )


# --------------------------------------------------
# 4. MASK AND PROBABILITY PNG VALIDATION
# --------------------------------------------------

for class_name in CLASSES:

    mask = decode_png(
        result["masks"][class_name]
    )

    probability = decode_png(
        result["probabilities"][class_name]
    )

    check(
        mask.shape == (256, 256),
        f"{class_name}: mask dimensions"
    )

    check(
        probability.shape == (256, 256),
        f"{class_name}: probability dimensions"
    )

    check(
        set(np.unique(mask)).issubset({0, 255}),
        f"{class_name}: binary mask values"
    )


# --------------------------------------------------
# 5. CONFIDENCE AND UNCERTAINTY
# --------------------------------------------------

for class_name in CLASSES:

    confidence = result["confidence"][class_name]

    uncertainty = result["uncertainty"][class_name]

    if confidence is not None:

        check(
            0 <= confidence <= 1,
            f"{class_name}: confidence range"
        )

    check(
        0 <= uncertainty <= 1,
        f"{class_name}: uncertainty range"
    )


# --------------------------------------------------
# 6. IMAGE UNCERTAINTY
# --------------------------------------------------

image_uncertainty = result["image_uncertainty"]

check(
    0 <= image_uncertainty <= 1,
    "Image uncertainty range"
)


# --------------------------------------------------
# 7. FINAL RESULT
# --------------------------------------------------

print("\n--------------------------------")
print("ALL INTEGRATION TESTS PASSED!")
print("--------------------------------")
