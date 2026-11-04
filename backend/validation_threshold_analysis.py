
from pathlib import Path

import numpy as np
import pandas as pd

from app.inference import ChestXrayInference, CLASS_NAMES


# =====================================================
# 1. PATHS
# =====================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "validation_manifest.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "images_256"
)

MASK_DIR = (
    PROJECT_ROOT / "data" / "processed" / "masks_256"
)

DICOM_DIR = (
    PROJECT_ROOT / "data" / "images" / "train" / "vindr_300"
)

OUTPUT_DIR = (
    BACKEND_DIR / "outputs" / "validation_threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]


# =====================================================
# 2. LOAD VALIDATION MANIFEST
# =====================================================

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != 43:
    raise ValueError(
        f"Expected 43 validation images, found {len(manifest)}"
    )

print("=" * 65)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 65)

print("Validation images:", len(manifest))
print("Thresholds:", THRESHOLDS)


# =====================================================
# 3. LOAD MODEL
# =====================================================

print("\nLoading Original U-Net...")

inference = ChestXrayInference()

print("Model loaded on:", inference.device)


# =====================================================
# 4. RUN INFERENCE ONCE PER IMAGE
# =====================================================

probability_cache = {}
ground_truth_cache = {}

for index, row in manifest.iterrows():

    image_id = str(row["image_id"])

    print(
        f"\n[{index + 1}/43] Processing {image_id}"
    )

    # Find DICOM
    dicom_path = DICOM_DIR / f"{image_id}.dicom"

    if not dicom_path.exists():
        dicom_path = DICOM_DIR / f"{image_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM not found: {image_id}"
        )

    # Find ground-truth mask
    mask_path = MASK_DIR / f"{image_id}.npy"

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    gt = np.load(mask_path)

    if gt.shape != (256, 256, 5):
        raise ValueError(
            f"Invalid mask shape for {image_id}: {gt.shape}"
        )

    gt = (gt > 0).astype(np.uint8)

    # Run model
    result = inference.predict(dicom_path)

    probabilities = result["probabilities"]

    if probabilities.shape != (5, 256, 256):
        raise ValueError(
            f"Unexpected prediction shape: {probabilities.shape}"
        )

    probability_cache[image_id] = probabilities.astype(
        np.float32
    )

    ground_truth_cache[image_id] = gt


print("\nAll validation predictions cached.")


# =====================================================
# 5. METRIC FUNCTIONS
# =====================================================

def calculate_metrics(tp, fp, fn, tn):

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

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0 else np.nan
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else np.nan
    )

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice,
        "iou": iou,
    }


# =====================================================
# 6. THRESHOLD SWEEP
# =====================================================

sweep_rows = []
image_rows = []

for class_index, class_name in enumerate(CLASS_NAMES):

    print(f"\nAnalyzing: {class_name}")

    for threshold in THRESHOLDS:

        pixel_tp = 0
        pixel_fp = 0
        pixel_fn = 0
        pixel_tn = 0

        image_tp = 0
        image_fp = 0
        image_fn = 0
        image_tn = 0

        predicted_areas = []

        for image_id in probability_cache:

            probability = probability_cache[image_id][
                class_index
            ]

            gt = ground_truth_cache[image_id][
                :, :, class_index
            ]

            pred = (
                probability >= threshold
            ).astype(np.uint8)

            # Pixel-level confusion counts
            pixel_tp += int(
                np.logical_and(gt == 1, pred == 1).sum()
            )

            pixel_fp += int(
                np.logical_and(gt == 0, pred == 1).sum()
            )

            pixel_fn += int(
                np.logical_and(gt == 1, pred == 0).sum()
            )

            pixel_tn += int(
                np.logical_and(gt == 0, pred == 0).sum()
            )

            # Image-level presence based on any predicted pixel
            gt_present = bool(gt.sum() > 0)
            pred_present = bool(pred.sum() > 0)

            if gt_present and pred_present:
                image_tp += 1

            elif not gt_present and pred_present:
                image_fp += 1

            elif gt_present and not pred_present:
                image_fn += 1

            else:
                image_tn += 1

            predicted_areas.append(int(pred.sum()))

            image_rows.append({
                "image_id": image_id,
                "class": class_name,
                "threshold": threshold,
                "gt_pixels": int(gt.sum()),
                "predicted_pixels": int(pred.sum()),
                "gt_present": int(gt_present),
                "predicted_present": int(pred_present),
            })

        pixel_metrics = calculate_metrics(
            pixel_tp,
            pixel_fp,
            pixel_fn,
            pixel_tn
        )

        image_metrics = calculate_metrics(
            image_tp,
            image_fp,
            image_fn,
            image_tn
        )

        sweep_rows.append({
            "class": class_name,
            "threshold": threshold,

            "pixel_TP": pixel_tp,
            "pixel_FP": pixel_fp,
            "pixel_FN": pixel_fn,
            "pixel_TN": pixel_tn,

            "pixel_precision": pixel_metrics["precision"],
            "pixel_recall": pixel_metrics["recall"],
            "pixel_dice": pixel_metrics["dice"],
            "pixel_iou": pixel_metrics["iou"],

            "image_TP": image_tp,
            "image_FP": image_fp,
            "image_TN": image_tn,
            "image_FN": image_fn,

            "image_precision": image_metrics["precision"],
            "image_recall": image_metrics["recall"],
            "image_specificity": image_metrics["specificity"],

            "mean_predicted_pixels": np.mean(predicted_areas),
        })


# =====================================================
# 7. CREATE DATAFRAMES
# =====================================================

sweep_df = pd.DataFrame(sweep_rows)
image_df = pd.DataFrame(image_rows)


# =====================================================
# 8. SELECT BEST DICE THRESHOLD
# =====================================================

best_rows = []

for class_name in CLASS_NAMES:

    class_df = sweep_df[
        sweep_df["class"] == class_name
    ].copy()

    valid_df = class_df.dropna(
        subset=["pixel_dice"]
    )

    if valid_df.empty:
        continue

    best_row = valid_df.loc[
        valid_df["pixel_dice"].idxmax()
    ]

    best_rows.append({
        "class": class_name,
        "best_dice_threshold": best_row["threshold"],
        "best_pixel_dice": best_row["pixel_dice"],
        "pixel_precision": best_row["pixel_precision"],
        "pixel_recall": best_row["pixel_recall"],
        "image_specificity": best_row["image_specificity"],
        "image_false_positives": best_row["image_FP"],
        "mean_predicted_pixels": best_row["mean_predicted_pixels"],
    })

best_df = pd.DataFrame(best_rows)


# =====================================================
# 9. SAVE REPORTS
# =====================================================

sweep_path = OUTPUT_DIR / "threshold_sweep.csv"
image_path = OUTPUT_DIR / "threshold_image_results.csv"
best_path = OUTPUT_DIR / "best_dice_thresholds.csv"

sweep_df.to_csv(sweep_path, index=False)
image_df.to_csv(image_path, index=False)
best_df.to_csv(best_path, index=False)


# =====================================================
# 10. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 75)
print("THRESHOLD SWEEP RESULTS")
print("=" * 75)

for class_name in CLASS_NAMES:

    print(f"\n{class_name}")

    display = sweep_df[
        sweep_df["class"] == class_name
    ][[
        "threshold",
        "pixel_precision",
        "pixel_recall",
        "pixel_dice",
        "image_specificity",
        "image_FP",
        "mean_predicted_pixels",
    ]]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print("\n" + "=" * 75)
print("BEST DICE THRESHOLDS ON VALIDATION SET")
print("=" * 75)

print(
    best_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nAnalysis completed.")

print("\nSaved files:")
print(sweep_path)
print(image_path)
print(best_path)
