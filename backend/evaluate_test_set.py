
from pathlib import Path
import csv

import numpy as np
import pandas as pd

from app.inference import (
    ChestXrayInference,
    CLASS_NAMES,
    THRESHOLDS,
)


# ==================================================
# 1. PATHS
# ==================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"
)

MASK_DIR = (
    PROJECT_ROOT / "data" / "processed" / "masks_256"
)

DICOM_DIR = (
    PROJECT_ROOT / "data" / "images" / "train" / "vindr_300"
)

OUTPUT_DIR = BACKEND_DIR / "outputs" / "full_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TEST MANIFEST
# ==================================================

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != 44:
    raise ValueError(
        f"Expected 44 test images, found {len(manifest)}"
    )

print("=" * 60)
print("FULL TEST SET EVALUATION")
print("=" * 60)

print("Test images:", len(manifest))
print("Classes:", CLASS_NAMES)


# ==================================================
# 3. LOAD MODEL
# ==================================================

print("\nLoading Original U-Net...")

inference = ChestXrayInference()

print("Model ready.")
print("Device:", inference.device)


# ==================================================
# 4. METRIC HELPERS
# ==================================================

def calculate_dice(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    denominator = gt.sum() + pred.sum()

    if denominator == 0:
        return np.nan

    intersection = np.logical_and(gt, pred).sum()

    return float(
        2 * intersection / denominator
    )


def calculate_iou(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return np.nan

    intersection = np.logical_and(gt, pred).sum()

    return float(intersection / union)


# ==================================================
# 5. EVALUATE ALL TEST IMAGES
# ==================================================

image_results = []

# Confusion matrix counts for each class
class_counts = {
    name: {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
    }
    for name in CLASS_NAMES
}

# Store image-level Dice only where GT is positive
positive_image_dice = {
    name: [] for name in CLASS_NAMES
}

# Pixel-level aggregate counts
pixel_counts = {
    name: {
        "TP": 0,
        "FP": 0,
        "FN": 0,
    }
    for name in CLASS_NAMES
}

for image_number, row in manifest.iterrows():

    image_id = str(row["image_id"])

    print(
        f"\n[{image_number + 1}/{len(manifest)}] "
        f"Processing {image_id}"
    )

    # ----------------------------------------------
    # Locate original DICOM
    # ----------------------------------------------

    dicom_path = DICOM_DIR / f"{image_id}.dicom"

    if not dicom_path.exists():
        dicom_path = DICOM_DIR / f"{image_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"Missing DICOM for image: {image_id}"
        )

    # ----------------------------------------------
    # Load ground-truth mask
    # ----------------------------------------------

    mask_path = MASK_DIR / f"{image_id}.npy"

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    gt_masks = np.load(mask_path)

    if gt_masks.shape != (256, 256, 5):
        raise ValueError(
            f"Invalid mask shape for {image_id}: "
            f"{gt_masks.shape}"
        )

    gt_masks = (gt_masks > 0).astype(np.uint8)

    # ----------------------------------------------
    # Run inference
    # ----------------------------------------------

    result = inference.predict(dicom_path)

    image_record = {
        "image_id": image_id,
    }

    # ----------------------------------------------
    # Evaluate each class
    # ----------------------------------------------

    for class_index, class_name in enumerate(CLASS_NAMES):

        gt = gt_masks[:, :, class_index]

        pred = result["masks"][class_name]

        pred = (pred > 0).astype(np.uint8)

        # Pixel-level counts
        tp_pixels = int(
            np.logical_and(gt == 1, pred == 1).sum()
        )

        fp_pixels = int(
            np.logical_and(gt == 0, pred == 1).sum()
        )

        fn_pixels = int(
            np.logical_and(gt == 1, pred == 0).sum()
        )

        pixel_counts[class_name]["TP"] += tp_pixels
        pixel_counts[class_name]["FP"] += fp_pixels
        pixel_counts[class_name]["FN"] += fn_pixels

        # Per-image Dice and IoU
        dice = calculate_dice(gt, pred)
        iou = calculate_iou(gt, pred)

        if gt.sum() > 0:
            positive_image_dice[class_name].append(dice)

        # Image-level presence
        gt_present = int(gt.sum() > 0)
        pred_present = int(pred.sum() > 0)

        if gt_present and pred_present:
            category = "TP"
        elif not gt_present and pred_present:
            category = "FP"
        elif not gt_present and not pred_present:
            category = "TN"
        else:
            category = "FN"

        class_counts[class_name][category] += 1

        # Record per-image results
        safe_name = class_name.replace(" ", "_")

        image_record[f"{safe_name}_gt_present"] = gt_present
        image_record[f"{safe_name}_pred_present"] = pred_present
        image_record[f"{safe_name}_image_result"] = category
        image_record[f"{safe_name}_gt_pixels"] = int(gt.sum())
        image_record[f"{safe_name}_pred_pixels"] = int(pred.sum())
        image_record[f"{safe_name}_dice"] = dice
        image_record[f"{safe_name}_iou"] = iou

    image_results.append(image_record)

    print("Completed:", image_id)


# ==================================================
# 6. IMAGE-LEVEL CLASSIFICATION SUMMARY
# ==================================================

summary_rows = []

for class_name in CLASS_NAMES:

    counts = class_counts[class_name]

    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0 else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn > 0 else np.nan
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
    )

    mean_positive_dice = (
        np.nanmean(positive_image_dice[class_name])
        if positive_image_dice[class_name]
        else np.nan
    )

    # Aggregate pixel-level Dice and IoU
    p = pixel_counts[class_name]

    pixel_tp = p["TP"]
    pixel_fp = p["FP"]
    pixel_fn = p["FN"]

    aggregate_dice = (
        2 * pixel_tp /
        (2 * pixel_tp + pixel_fp + pixel_fn)
        if 2 * pixel_tp + pixel_fp + pixel_fn > 0
        else np.nan
    )

    aggregate_iou = (
        pixel_tp /
        (pixel_tp + pixel_fp + pixel_fn)
        if pixel_tp + pixel_fp + pixel_fn > 0
        else np.nan
    )

    summary_rows.append({
        "class": class_name,
        "threshold": THRESHOLDS[class_name],

        "TP_images": tp,
        "FP_images": fp,
        "TN_images": tn,
        "FN_images": fn,

        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,

        "mean_positive_image_dice": mean_positive_dice,
        "aggregate_pixel_dice": aggregate_dice,
        "aggregate_pixel_iou": aggregate_iou,

        "positive_images": tp + fn,
        "negative_images": tn + fp,
    })


# ==================================================
# 7. SAVE CSV REPORTS
# ==================================================

image_df = pd.DataFrame(image_results)

summary_df = pd.DataFrame(summary_rows)

image_csv = OUTPUT_DIR / "per_image_results.csv"
summary_csv = OUTPUT_DIR / "per_class_summary.csv"

image_df.to_csv(image_csv, index=False)
summary_df.to_csv(summary_csv, index=False)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 70)
print("IMAGE-LEVEL CLASSIFICATION RESULTS")
print("=" * 70)

display_columns = [
    "class",
    "TP_images",
    "FP_images",
    "TN_images",
    "FN_images",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
]

print(
    summary_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("SEGMENTATION RESULTS")
print("=" * 70)

segmentation_columns = [
    "class",
    "mean_positive_image_dice",
    "aggregate_pixel_dice",
    "aggregate_pixel_iou",
]

print(
    summary_df[segmentation_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nSaved reports:")
print(image_csv)
print(summary_csv)
