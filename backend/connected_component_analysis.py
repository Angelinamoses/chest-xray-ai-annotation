
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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_manifest.csv"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "masks_256"
)

DICOM_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "train"
    / "vindr_300"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "outputs"
    / "connected_component_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Frozen thresholds selected using validation experiments
THRESHOLDS = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}

EXPECTED_VALIDATION_IMAGES = 43
EXPECTED_IMAGE_SIZE = (256, 256)

COMPONENT_CUTOFFS = [10, 25, 50, 100, 250]

# 8-connectivity
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


# =====================================================
# 2. CONFIGURATION VALIDATION
# =====================================================

if len(CLASS_NAMES) != 5:
    raise ValueError(
        f"Expected 5 classes, found {len(CLASS_NAMES)}"
    )

if set(THRESHOLDS.keys()) != set(CLASS_NAMES):
    raise ValueError(
        "Threshold configuration does not match CLASS_NAMES."
    )

for class_name, threshold in THRESHOLDS.items():
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid threshold for {class_name}: {threshold}"
        )


# =====================================================
# 3. LOAD VALIDATION MANIFEST
# =====================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Validation manifest not found: {MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

if "image_id" not in manifest.columns:
    raise ValueError(
        "Manifest must contain an image_id column."
    )

if len(manifest) != EXPECTED_VALIDATION_IMAGES:
    raise ValueError(
        f"Expected {EXPECTED_VALIDATION_IMAGES} validation "
        f"images, found {len(manifest)}."
    )

manifest["image_id"] = manifest["image_id"].astype(str)

if manifest["image_id"].duplicated().any():
    raise ValueError(
        "Duplicate image IDs found in validation manifest."
    )

print("=" * 75)
print("CONNECTED-COMPONENT ANALYSIS")
print("=" * 75)

print("Validation images:", len(manifest))
print("Classes:", CLASS_NAMES)
print("Thresholds:", THRESHOLDS)


# =====================================================
# 4. LOAD MODEL
# =====================================================

print("\nLoading Original U-Net...")

inference = ChestXrayInference()

print("Model loaded successfully.")
print("Device:", inference.device)


# =====================================================
# 5. METRIC FUNCTIONS
# =====================================================

def calculate_segmentation_metrics(gt, pred):
    """
    Calculate segmentation metrics for one image.

    Both inputs must have identical spatial dimensions.
    Empty GT and empty prediction produce NaN Dice/IoU.
    """

    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    if gt.shape != pred.shape:
        raise ValueError(
            f"GT shape {gt.shape} does not match "
            f"prediction shape {pred.shape}."
        )

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else np.nan
    )

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
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


def calculate_pooled_metrics(tp, fp, fn):
    """
    Calculate pooled metrics directly from accumulated
    pixel-level TP, FP and FN counts.

    This avoids constructing artificial arrays.
    """

    tp = int(tp)
    fp = int(fp)
    fn = int(fn)

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    return {
        "pixel_precision": (
            tp / (tp + fp)
            if tp + fp > 0
            else np.nan
        ),

        "pixel_recall": (
            tp / (tp + fn)
            if tp + fn > 0
            else np.nan
        ),

        "pixel_dice": (
            2 * tp / dice_denominator
            if dice_denominator > 0
            else np.nan
        ),

        "pixel_iou": (
            tp / iou_denominator
            if iou_denominator > 0
            else np.nan
        ),
    }


def calculate_image_metrics(gt, pred):
    """
    Image-level presence classification relative to
    the bounding-box-derived pseudo-mask.
    """

    gt_present = bool(np.any(gt))
    pred_present = bool(np.any(pred))

    if gt_present and pred_present:
        return "TP"

    if not gt_present and pred_present:
        return "FP"

    if not gt_present and not pred_present:
        return "TN"

    return "FN"


# =====================================================
# 6. CONNECTED-COMPONENT FUNCTIONS
# =====================================================

def get_component_statistics(mask):
    """
    Identify 8-connected components and calculate
    their areas.
    """

    mask = np.asarray(mask).astype(bool)

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
        if total_area > 0
        else 0.0
    )

    return {
        "component_count": int(number_of_components),
        "largest_component_area": largest_area,
        "largest_component_fraction": largest_fraction,
        "component_areas": areas.tolist(),
        "labels": labeled,
    }


def keep_largest_component(labels):
    """
    Retain only the largest connected component.
    """

    component_ids, counts = np.unique(
        labels[labels > 0],
        return_counts=True
    )

    if len(component_ids) == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    largest_id = component_ids[np.argmax(counts)]

    return (labels == largest_id).astype(np.uint8)


def remove_small_components(labels, minimum_size):
    """
    Retain components whose pixel area is at least
    minimum_size.
    """

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
# 7. EVALUATION STORAGE
# =====================================================

component_rows = []
image_rows = []

METHODS = (
    ["baseline", "largest_component"]
    + [
        f"minimum_component_{cutoff}"
        for cutoff in COMPONENT_CUTOFFS
    ]
)

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
# 8. RUN INFERENCE AND COMPONENT ANALYSIS
# =====================================================

for image_index, row in enumerate(
    manifest.itertuples(index=False),
    start=1
):

    image_id = str(row.image_id)

    print(
        f"\n[{image_index}/{len(manifest)}] "
        f"Processing {image_id}"
    )

    # -------------------------------------------------
    # Locate DICOM
    # -------------------------------------------------

    dicom_path = DICOM_DIR / f"{image_id}.dicom"

    if not dicom_path.exists():
        dicom_path = DICOM_DIR / f"{image_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM not found for image: {image_id}"
        )

    # -------------------------------------------------
    # Load pseudo-mask
    # -------------------------------------------------

    gt_path = MASK_DIR / f"{image_id}.npy"

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Mask not found for image: {image_id}"
        )

    gt_all = np.load(gt_path)

    expected_mask_shape = (
        EXPECTED_IMAGE_SIZE[0],
        EXPECTED_IMAGE_SIZE[1],
        len(CLASS_NAMES)
    )

    if gt_all.shape != expected_mask_shape:
        raise ValueError(
            f"Invalid mask shape for {image_id}: "
            f"{gt_all.shape}. Expected {expected_mask_shape}."
        )

    gt_all = (gt_all > 0).astype(np.uint8)

    # -------------------------------------------------
    # Model inference
    # -------------------------------------------------

    result = inference.predict(dicom_path)

    probabilities = np.asarray(
        result["probabilities"]
    )

    expected_prediction_shape = (
        len(CLASS_NAMES),
        EXPECTED_IMAGE_SIZE[0],
        EXPECTED_IMAGE_SIZE[1]
    )

    if probabilities.shape != expected_prediction_shape:
        raise ValueError(
            f"Invalid probability shape for {image_id}: "
            f"{probabilities.shape}. "
            f"Expected {expected_prediction_shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Non-finite probabilities found for {image_id}."
        )

    # -------------------------------------------------
    # Class-wise analysis
    # -------------------------------------------------

    for class_index, class_name in enumerate(CLASS_NAMES):

        threshold = THRESHOLDS[class_name]

        gt = gt_all[:, :, class_index]

        probability = probabilities[class_index]

        # Original thresholded prediction
        baseline = (
            probability >= threshold
        ).astype(np.uint8)

        # Connected-component statistics
        stats = get_component_statistics(baseline)

        labels = stats["labels"]

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

        # -------------------------------------------------
        # Filtering strategies
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Evaluate each strategy
        # -------------------------------------------------

        for method, pred in filtered_predictions.items():

            pixel_metrics = calculate_segmentation_metrics(
                gt,
                pred
            )

            image_result = calculate_image_metrics(
                gt,
                pred
            )

            counts = aggregate[class_name][method]

            counts["TP_pixels"] += (
                pixel_metrics["TP_pixels"]
            )

            counts["FP_pixels"] += (
                pixel_metrics["FP_pixels"]
            )

            counts["FN_pixels"] += (
                pixel_metrics["FN_pixels"]
            )

            counts[f"{image_result}_images"] += 1

            image_rows.append({
                "image_id": image_id,
                "class": class_name,
                "method": method,
                "threshold": threshold,
                "gt_pixels": int(gt.sum()),
                "predicted_pixels": int(pred.sum()),
                "image_result": image_result,
                **pixel_metrics,
            })


# =====================================================
# 9. CREATE CORRECTED AGGREGATE SUMMARY
# =====================================================

summary_rows = []

for class_name in CLASS_NAMES:

    for method in METHODS:

        counts = aggregate[class_name][method]

        tp = counts["TP_pixels"]
        fp = counts["FP_pixels"]
        fn = counts["FN_pixels"]

        # Correct pooled pixel-level metrics
        pixel_metrics = calculate_pooled_metrics(
            tp,
            fp,
            fn
        )

        image_tp = counts["TP_images"]
        image_fp = counts["FP_images"]
        image_tn = counts["TN_images"]
        image_fn = counts["FN_images"]

        image_precision = (
            image_tp / (image_tp + image_fp)
            if image_tp + image_fp > 0
            else np.nan
        )

        image_recall = (
            image_tp / (image_tp + image_fn)
            if image_tp + image_fn > 0
            else np.nan
        )

        image_specificity = (
            image_tn / (image_tn + image_fp)
            if image_tn + image_fp > 0
            else np.nan
        )

        image_f1 = (
            2 * image_precision * image_recall
            / (image_precision + image_recall)
            if (
                np.isfinite(image_precision)
                and np.isfinite(image_recall)
                and image_precision + image_recall > 0
            )
            else np.nan
        )

        summary_rows.append({
            "class": class_name,
            "method": method,

            # Pixel-level confusion counts
            "pixel_TP": tp,
            "pixel_FP": fp,
            "pixel_FN": fn,

            # Correct pooled segmentation metrics
            **pixel_metrics,

            # Image-level confusion counts
            "image_TP": image_tp,
            "image_FP": image_fp,
            "image_TN": image_tn,
            "image_FN": image_fn,

            # Image-level metrics
            "image_precision": image_precision,
            "image_recall": image_recall,
            "image_specificity": image_specificity,
            "image_f1": image_f1,
        })


# =====================================================
# 10. SAVE REPORTS
# =====================================================

component_df = pd.DataFrame(component_rows)

image_df = pd.DataFrame(image_rows)

summary_df = pd.DataFrame(summary_rows)

component_path = (
    OUTPUT_DIR / "component_statistics.csv"
)

image_path = (
    OUTPUT_DIR / "filtered_image_results.csv"
)

summary_path = (
    OUTPUT_DIR / "component_filter_summary.csv"
)

component_df.to_csv(
    component_path,
    index=False
)

image_df.to_csv(
    image_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# =====================================================
# 11. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 85)
print("CORRECTED COMPONENT FILTERING SUMMARY")
print("=" * 85)

for class_name in CLASS_NAMES:

    print(f"\n{class_name}")

    subset = summary_df[
        summary_df["class"] == class_name
    ][[
        "method",
        "pixel_dice",
        "pixel_iou",
        "pixel_precision",
        "pixel_recall",
        "image_precision",
        "image_recall",
        "image_specificity",
        "image_f1",
        "image_FP",
    ]]

    print(
        subset.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# =====================================================
# 12. FINAL VALIDATION
# =====================================================

expected_component_rows = (
    len(manifest) * len(CLASS_NAMES)
)

expected_image_rows = (
    len(manifest) * len(CLASS_NAMES) * len(METHODS)
)

if len(component_df) != expected_component_rows:
    raise RuntimeError(
        "Unexpected number of component-statistics rows."
    )

if len(image_df) != expected_image_rows:
    raise RuntimeError(
        "Unexpected number of per-image evaluation rows."
    )

if len(summary_df) != len(CLASS_NAMES) * len(METHODS):
    raise RuntimeError(
        "Unexpected number of aggregate summary rows."
    )

print("\n" + "=" * 85)
print("CONNECTED-COMPONENT ANALYSIS COMPLETED")
print("=" * 85)

print("\nValidation images:", len(manifest))
print("Classes:", len(CLASS_NAMES))
print("Methods:", len(METHODS))

print("\nSaved reports:")

print(component_path)
print(image_path)
print(summary_path)

print("\nAll output validation checks passed.")
