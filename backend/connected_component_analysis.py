
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from app.inference import ChestXrayInference, CLASS_NAMES


# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "validation_manifest.csv"
)

MASK_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "masks_256"
)

DICOM_DIR = (
    PROJECT_ROOT / "data" / "images"
    / "train" / "vindr_300"
)

OUTPUT_DIR = (
    BACKEND_DIR / "outputs"
    / "connected_component_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen probability thresholds
THRESHOLDS = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}

# Minimum connected-component sizes
COMPONENT_CUTOFFS = [10, 25, 50, 100, 250]

# 8-connectivity
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


# =====================================================
# 2. LOAD VALIDATION MANIFEST
# =====================================================

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != 43:
    raise ValueError(
        f"Expected 43 validation images, got {len(manifest)}"
    )

print("=" * 70)
print("CONNECTED-COMPONENT ANALYSIS")
print("=" * 70)

print("Validation images:", len(manifest))
print("Classes:", CLASS_NAMES)


# =====================================================
# 3. LOAD MODEL
# =====================================================

print("\nLoading Original U-Net...")

inference = ChestXrayInference()

print("Model loaded on:", inference.device)


# =====================================================
# 4. METRIC FUNCTIONS
# =====================================================

def calculate_segmentation_metrics(gt, pred):

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0 else np.nan
    )

    iou = (
        tp / iou_denominator
        if iou_denominator > 0 else np.nan
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else np.nan
    )

    return {
        "TP_pixels": tp,
        "FP_pixels": fp,
        "FN_pixels": fn,
        "dice": dice,
        "iou": iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
    }


def calculate_image_metrics(gt, pred):

    gt_present = bool(gt.sum() > 0)
    pred_present = bool(pred.sum() > 0)

    if gt_present and pred_present:
        return "TP"

    if not gt_present and pred_present:
        return "FP"

    if not gt_present and not pred_present:
        return "TN"

    return "FN"


# =====================================================
# 5. CONNECTED-COMPONENT HELPERS
# =====================================================

def get_component_statistics(mask):

    labeled, number_of_components = ndimage.label(
        mask,
        structure=CONNECTIVITY
    )

    if number_of_components == 0:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_fraction": 0.0,
            "component_areas": [],
            "labels": labeled,
        }

    areas = np.bincount(
        labeled.ravel()
    )[1:]

    largest_area = int(areas.max())
    total_area = int(mask.sum())

    largest_fraction = (
        largest_area / total_area
        if total_area > 0 else 0.0
    )

    return {
        "component_count": int(number_of_components),
        "largest_component_area": largest_area,
        "largest_component_fraction": largest_fraction,
        "component_areas": areas.tolist(),
        "labels": labeled,
    }


def keep_largest_component(labels):

    component_ids, counts = np.unique(
        labels[labels > 0],
        return_counts=True
    )

    if len(component_ids) == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    largest_id = component_ids[np.argmax(counts)]

    return (labels == largest_id).astype(np.uint8)


def remove_small_components(labels, minimum_size):

    component_ids, counts = np.unique(
        labels[labels > 0],
        return_counts=True
    )

    result = np.zeros_like(labels, dtype=np.uint8)

    for component_id, area in zip(component_ids, counts):

        if area >= minimum_size:
            result[labels == component_id] = 1

    return result


# =====================================================
# 6. EVALUATION STORAGE
# =====================================================

component_rows = []
image_rows = []

# Methods compared
METHODS = (
    ["baseline", "largest_component"]
    + [f"minimum_component_{x}" for x in COMPONENT_CUTOFFS]
)

# Aggregate metrics per class and method
aggregate = {
    class_name: {
        method: {
            "TP_pixels": 0,
            "FP_pixels": 0,
            "FN_pixels": 0,
            "TP_images": 0,
            "FP_images": 0,
            "TN_images": 0,
            "FN_images": 0,
        }
        for method in METHODS
    }
    for class_name in CLASS_NAMES
}


# =====================================================
# 7. RUN INFERENCE AND COMPONENT ANALYSIS
# =====================================================

for image_index, row in manifest.iterrows():

    image_id = str(row["image_id"])

    print(
        f"\n[{image_index + 1}/43] Processing {image_id}"
    )

    # Locate DICOM
    dicom_path = DICOM_DIR / f"{image_id}.dicom"

    if not dicom_path.exists():
        dicom_path = DICOM_DIR / f"{image_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(dicom_path)

    # Load pseudo-mask
    gt_path = MASK_DIR / f"{image_id}.npy"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt_all = np.load(gt_path)

    if gt_all.shape != (256, 256, 5):
        raise ValueError(
            f"Invalid mask shape: {gt_all.shape}"
        )

    gt_all = (gt_all > 0).astype(np.uint8)

    # Inference
    result = inference.predict(dicom_path)

    probabilities = result["probabilities"]

    for class_index, class_name in enumerate(CLASS_NAMES):

        threshold = THRESHOLDS[class_name]

        gt = gt_all[:, :, class_index]

        probability = probabilities[class_index]

        # Original thresholded prediction
        baseline = (
            probability >= threshold
        ).astype(np.uint8)

        stats = get_component_statistics(baseline)

        labels = stats["labels"]

        # Component distribution
        component_rows.append({
            "image_id": image_id,
            "class": class_name,
            "threshold": threshold,
            "ground_truth_pixels": int(gt.sum()),
            "predicted_pixels": int(baseline.sum()),
            "component_count": stats["component_count"],
            "largest_component_area": (
                stats["largest_component_area"]
            ),
            "largest_component_fraction": (
                stats["largest_component_fraction"]
            ),
            "component_areas": ";".join(
                map(str, stats["component_areas"])
            ),
        })

        # Filtering strategies
        filtered_predictions = {
            "baseline": baseline,
            "largest_component": keep_largest_component(
                labels
            ),
        }

        for cutoff in COMPONENT_CUTOFFS:

            filtered_predictions[
                f"minimum_component_{cutoff}"
            ] = remove_small_components(
                labels,
                cutoff
            )

        # Evaluate every strategy
        for method, pred in filtered_predictions.items():

            pixel_metrics = calculate_segmentation_metrics(
                gt, pred
            )

            image_result = calculate_image_metrics(
                gt, pred
            )

            counts = aggregate[class_name][method]

            counts["TP_pixels"] += pixel_metrics["TP_pixels"]
            counts["FP_pixels"] += pixel_metrics["FP_pixels"]
            counts["FN_pixels"] += pixel_metrics["FN_pixels"]

            counts[f"{image_result}_images"] += 1

            image_rows.append({
                "image_id": image_id,
                "class": class_name,
                "method": method,
                "gt_pixels": int(gt.sum()),
                "predicted_pixels": int(pred.sum()),
                "image_result": image_result,
                **pixel_metrics,
            })


# =====================================================
# 8. CREATE SUMMARY
# =====================================================

summary_rows = []

for class_name in CLASS_NAMES:

    for method in METHODS:

        counts = aggregate[class_name][method]

        tp = counts["TP_pixels"]
        fp = counts["FP_pixels"]
        fn = counts["FN_pixels"]

        pixel_metrics = calculate_segmentation_metrics(
            np.array([1] * (tp + fn) + [0] * fp),
            np.array([1] * tp + [0] * (fp + fn))
        )

        image_tp = counts["TP_images"]
        image_fp = counts["FP_images"]
        image_tn = counts["TN_images"]
        image_fn = counts["FN_images"]

        image_precision = (
            image_tp / (image_tp + image_fp)
            if image_tp + image_fp > 0 else np.nan
        )

        image_recall = (
            image_tp / (image_tp + image_fn)
            if image_tp + image_fn > 0 else np.nan
        )

        image_specificity = (
            image_tn / (image_tn + image_fp)
            if image_tn + image_fp > 0 else np.nan
        )

        summary_rows.append({
            "class": class_name,
            "method": method,

            "pixel_TP": tp,
            "pixel_FP": fp,
            "pixel_FN": fn,

            "pixel_precision": pixel_metrics["pixel_precision"],
            "pixel_recall": pixel_metrics["pixel_recall"],
            "pixel_dice": pixel_metrics["dice"],
            "pixel_iou": pixel_metrics["iou"],

            "image_TP": image_tp,
            "image_FP": image_fp,
            "image_TN": image_tn,
            "image_FN": image_fn,

            "image_precision": image_precision,
            "image_recall": image_recall,
            "image_specificity": image_specificity,
        })


# =====================================================
# 9. SAVE REPORTS
# =====================================================

component_df = pd.DataFrame(component_rows)
image_df = pd.DataFrame(image_rows)
summary_df = pd.DataFrame(summary_rows)

component_path = OUTPUT_DIR / "component_statistics.csv"
image_path = OUTPUT_DIR / "filtered_image_results.csv"
summary_path = OUTPUT_DIR / "component_filter_summary.csv"

component_df.to_csv(component_path, index=False)
image_df.to_csv(image_path, index=False)
summary_df.to_csv(summary_path, index=False)


# =====================================================
# 10. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 75)
print("COMPONENT FILTERING SUMMARY")
print("=" * 75)

for class_name in CLASS_NAMES:

    print(f"\n{class_name}")

    subset = summary_df[
        summary_df["class"] == class_name
    ][[
        "method",
        "pixel_dice",
        "pixel_precision",
        "pixel_recall",
        "image_precision",
        "image_recall",
        "image_specificity",
        "image_FP",
    ]]

    print(
        subset.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

print("\n" + "=" * 75)
print("COMPONENT ANALYSIS COMPLETED")
print("=" * 75)

print("\nSaved reports:")
print(component_path)
print(image_path)
print(summary_path)
