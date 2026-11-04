
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from app.inference import ChestXrayInference, CLASS_NAMES


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"
)

IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images_256"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks_256"

OUTPUT_DIR = BACKEND_DIR / "outputs" / "test_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}

CLASS_COLORS = {
    "Aortic enlargement": (255, 60, 60),
    "Cardiomegaly": (50, 220, 80),
    "Pleural thickening": (50, 150, 255),
    "Pulmonary fibrosis": (255, 210, 40),
    "Lung Opacity": (220, 60, 220),
}

ALPHA = 0.45


# --------------------------------------------------
# SELECT A TEST IMAGE
# --------------------------------------------------

with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    test_rows = list(reader)

if not test_rows:
    raise ValueError("Test manifest is empty.")

# First test image from the manifest
selected_row = test_rows[0]
image_id = selected_row["image_id"]

image_path = IMAGE_DIR / f"{image_id}.npy"
mask_path = MASK_DIR / f"{image_id}.npy"

if not image_path.exists():
    raise FileNotFoundError(image_path)

if not mask_path.exists():
    raise FileNotFoundError(mask_path)

print("Selected test image:", image_id)


# --------------------------------------------------
# LOAD IMAGE AND GROUND-TRUTH MASK
# --------------------------------------------------

image = np.load(image_path).astype(np.float32)
ground_truth = np.load(mask_path)

if image.shape != (256, 256):
    raise ValueError(f"Unexpected image shape: {image.shape}")

if ground_truth.shape != (256, 256, 5):
    raise ValueError(
        f"Unexpected ground-truth shape: {ground_truth.shape}"
    )

ground_truth = (ground_truth > 0).astype(np.uint8)

image_uint8 = (
    np.clip(image, 0, 1) * 255
).round().astype(np.uint8)

background = np.stack(
    [image_uint8] * 3,
    axis=-1
)


# --------------------------------------------------
# LOAD MODEL AND RUN INFERENCE
# --------------------------------------------------

print("\nLoading trained model...")

inference = ChestXrayInference()

# Use the original DICOM to ensure inference uses
# the same preprocessing pipeline as the API.
dicom_dir = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "train"
    / "vindr_300"
)

dicom_path = dicom_dir / f"{image_id}.dicom"

if not dicom_path.exists():
    dicom_path = dicom_dir / f"{image_id}.dcm"

if not dicom_path.exists():
    raise FileNotFoundError(
        f"Original DICOM not found for {image_id}"
    )

prediction_result = inference.predict(dicom_path)

print("Inference completed.")


# --------------------------------------------------
# VISUALIZATION HELPERS
# --------------------------------------------------

def create_overlay(image_rgb, mask, color, alpha=0.45):
    result = image_rgb.astype(np.float32).copy()

    color_array = np.array(color, dtype=np.float32)

    result[mask > 0] = (
        (1 - alpha) * result[mask > 0]
        + alpha * color_array
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def create_error_map(gt, pred):
    """
    Green = True Positive
    Red   = False Positive
    Blue  = False Negative
    Black = True Negative
    """

    error = np.zeros(
        (gt.shape[0], gt.shape[1], 3),
        dtype=np.uint8
    )

    true_positive = (gt == 1) & (pred == 1)
    false_positive = (gt == 0) & (pred == 1)
    false_negative = (gt == 1) & (pred == 0)

    error[true_positive] = [0, 255, 0]
    error[false_positive] = [255, 0, 0]
    error[false_negative] = [0, 100, 255]

    return error


# --------------------------------------------------
# CREATE FIVE-CLASS COMPARISON
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=5,
    ncols=6,
    figsize=(19, 17)
)

column_titles = [
    "Original X-ray",
    "Ground Truth",
    "Prediction",
    "Ground Truth Overlay",
    "Prediction Overlay",
    "Error Map",
]

for col, title in enumerate(column_titles):
    axes[0, col].set_title(
        title,
        fontsize=10,
        fontweight="bold"
    )


for class_index, class_name in enumerate(CLASS_NAMES):

    gt = ground_truth[:, :, class_index]

    probability = prediction_result[
        "probabilities"
    ][class_index]

    threshold = THRESHOLDS[class_name]

    pred = (
        probability >= threshold
    ).astype(np.uint8)

    color = CLASS_COLORS[class_name]

    gt_overlay = create_overlay(
        background,
        gt,
        color
    )

    pred_overlay = create_overlay(
        background,
        pred,
        color
    )

    error_map = create_error_map(gt, pred)

    panels = [
        image_uint8,
        gt * 255,
        pred * 255,
        gt_overlay,
        pred_overlay,
        error_map,
    ]

    for col, panel in enumerate(panels):
        ax = axes[class_index, col]

        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)

        ax.axis("off")

    axes[class_index, 0].set_ylabel(
        class_name,
        fontsize=10,
        fontweight="bold",
        rotation=90,
        labelpad=12
    )

    # Calculate pixel-level comparison statistics
    tp = int(np.logical_and(gt == 1, pred == 1).sum())
    fp = int(np.logical_and(gt == 0, pred == 1).sum())
    fn = int(np.logical_and(gt == 1, pred == 0).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    print(f"\n{class_name}")
    print(f"  Threshold: {threshold}")
    print(f"  Ground-truth pixels: {int(gt.sum())}")
    print(f"  Predicted pixels: {int(pred.sum())}")
    print(f"  True positives: {tp}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")
    print(f"  Dice: {dice:.4f}")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

fig.suptitle(
    f"Test Image: {image_id}\n"
    "Original U-Net: Prediction vs Ground-Truth Pseudo-Masks",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

comparison_path = OUTPUT_DIR / "five_class_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nCOMPARISON COMPLETED!")
print("Saved to:", comparison_path)
