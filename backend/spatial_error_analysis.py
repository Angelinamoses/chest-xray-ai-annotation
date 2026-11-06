
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.inference import ChestXrayInference, CLASS_NAMES
from app.preprocessing import preprocess_dicom


# =====================================================
# 1. CONFIGURATION
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
    / "spatial_error_analysis"
)

VISUAL_DIR = OUTPUT_DIR / "visualizations"
EXAMPLE_DIR = OUTPUT_DIR / "examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}

EXPECTED_IMAGES = 43
IMAGE_SIZE = 256

# Divide the 256 x 256 image into 4 x 4 regions.
GRID_SIZE = 4

# Pixels within 16 pixels of an image edge are considered
# part of the border band.
BORDER_WIDTH = 16

# Save the three images with the highest FP pixel counts
# for each class.
TOP_K_EXAMPLES = 3


# =====================================================
# 2. LOAD AND VALIDATE MANIFEST
# =====================================================

manifest = pd.read_csv(MANIFEST_PATH)

if "image_id" not in manifest.columns:
    raise ValueError("Manifest is missing image_id.")

if len(manifest) != EXPECTED_IMAGES:
    raise ValueError(
        f"Expected {EXPECTED_IMAGES} images, "
        f"found {len(manifest)}."
    )

manifest["image_id"] = manifest["image_id"].astype(str)

if manifest["image_id"].duplicated().any():
    raise ValueError("Duplicate image IDs found.")

if set(THRESHOLDS) != set(CLASS_NAMES):
    raise ValueError("Thresholds do not match model classes.")


# =====================================================
# 3. LOAD MODEL
# =====================================================

print("=" * 75)
print("SPATIAL FALSE-POSITIVE LOCALIZATION ANALYSIS")
print("=" * 75)

print("Validation images:", len(manifest))
print("Classes:", CLASS_NAMES)

inference = ChestXrayInference()

print("Model loaded on:", inference.device)


# =====================================================
# 4. SPATIAL HELPERS
# =====================================================

def get_border_mask(height, width, border_width):
    """Create a mask identifying image-border pixels."""

    border = np.zeros((height, width), dtype=bool)

    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    return border


def get_grid_counts(mask, grid_size=4):
    """
    Count positive pixels in each cell of a spatial grid.
    """

    height, width = mask.shape

    row_edges = np.linspace(
        0, height, grid_size + 1, dtype=int
    )

    col_edges = np.linspace(
        0, width, grid_size + 1, dtype=int
    )

    counts = np.zeros(
        (grid_size, grid_size),
        dtype=np.int64
    )

    for row in range(grid_size):
        for col in range(grid_size):

            region = mask[
                row_edges[row]:row_edges[row + 1],
                col_edges[col]:col_edges[col + 1]
            ]

            counts[row, col] = int(region.sum())

    return counts


def get_centroid(mask):
    """Return normalized centroid coordinates."""

    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return np.nan, np.nan

    center_y, center_x = coordinates.mean(axis=0)

    height, width = mask.shape

    return (
        float(center_y / height),
        float(center_x / width)
    )


def get_quadrant_counts(mask):
    """Calculate FP pixels in four image quadrants."""

    height, width = mask.shape

    mid_y = height // 2
    mid_x = width // 2

    return {
        "top_left": int(mask[:mid_y, :mid_x].sum()),
        "top_right": int(mask[:mid_y, mid_x:].sum()),
        "bottom_left": int(mask[mid_y:, :mid_x].sum()),
        "bottom_right": int(mask[mid_y:, mid_x:].sum()),
    }


# =====================================================
# 5. VISUALIZATION HELPERS
# =====================================================

def save_error_visualization(
    image,
    gt,
    pred,
    fp,
    fn,
    class_name,
    image_id
):
    """
    Save four-panel visualization:

    1. Original preprocessed X-ray
    2. Ground-truth pseudo-mask
    3. Model prediction
    4. Error map

    Error map:
    Green = TP
    Red = FP
    Blue = FN
    """

    tp = np.logical_and(gt, pred)
    fp = np.asarray(fp).astype(bool)
    fn = np.asarray(fn).astype(bool)

    error_map = np.zeros(
        (*gt.shape, 3),
        dtype=np.float32
    )

    error_map[tp] = [0.0, 0.8, 0.0]
    error_map[fp] = [1.0, 0.0, 0.0]
    error_map[fn] = [0.0, 0.3, 1.0]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 4)
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original X-ray")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Pseudo-mask")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")

    axes[3].imshow(error_map)
    axes[3].set_title(
        "Error map\nGreen: TP | Red: FP | Blue: FN"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{class_name} | {image_id}",
        fontsize=12
    )

    plt.tight_layout()

    safe_class_name = (
        class_name.lower()
        .replace(" ", "_")
    )

    output_path = (
        EXAMPLE_DIR
        / f"{safe_class_name}_{image_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_spatial_heatmap(grid, class_name):
    """Save aggregate 4 x 4 FP distribution heatmap."""

    safe_name = (
        class_name.lower()
        .replace(" ", "_")
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        grid,
        cmap="hot",
        interpolation="nearest"
    )

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            value = int(grid[row, col])

            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white"
            )

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    ax.set_xticklabels(
        ["Left 1", "Left 2", "Right 1", "Right 2"]
    )

    ax.set_yticklabels(
        ["Top 1", "Top 2", "Bottom 1", "Bottom 2"]
    )

    ax.set_title(
        f"False-positive spatial distribution\n"
        f"{class_name}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="False-positive pixel count"
    )

    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"{safe_name}_fp_heatmap.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 6. ANALYSIS STORAGE
# =====================================================

per_image_rows = []
grid_summary_rows = []

# Aggregate false-positive spatial grids per class.
class_grids = {
    class_name: np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=np.int64
    )
    for class_name in CLASS_NAMES
}

class_fp_totals = {
    class_name: 0
    for class_name in CLASS_NAMES
}

class_border_totals = {
    class_name: {
        "border_fp": 0,
        "interior_fp": 0,
    }
    for class_name in CLASS_NAMES
}

example_candidates = {
    class_name: []
    for class_name in CLASS_NAMES
}

border_mask = get_border_mask(
    IMAGE_SIZE,
    IMAGE_SIZE,
    BORDER_WIDTH
)


# =====================================================
# 7. PROCESS VALIDATION IMAGES
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

    if gt_all.shape != (IMAGE_SIZE, IMAGE_SIZE, 5):
        raise ValueError(
            f"Invalid mask shape for {image_id}: "
            f"{gt_all.shape}"
        )

    gt_all = (gt_all > 0)

    # Preprocess image for visualization.
    image_tensor, original_size, ds = preprocess_dicom(
        dicom_path,
        image_size=IMAGE_SIZE
    )

    display_image = (
        image_tensor[0, 0].cpu().numpy()
    )

    # Run inference.
    result = inference.predict(dicom_path)

    probabilities = np.asarray(
        result["probabilities"]
    )

    if probabilities.shape != (5, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    # -------------------------------------------------
    # Class-wise error analysis
    # -------------------------------------------------

    for class_index, class_name in enumerate(CLASS_NAMES):

        threshold = THRESHOLDS[class_name]

        gt = gt_all[:, :, class_index]

        probability = probabilities[class_index]

        pred = probability >= threshold

        tp = np.logical_and(gt, pred)
        fp = np.logical_and(~gt, pred)
        fn = np.logical_and(gt, ~pred)

        tp_count = int(tp.sum())
        fp_count = int(fp.sum())
        fn_count = int(fn.sum())

        gt_count = int(gt.sum())
        pred_count = int(pred.sum())

        # -------------------------------------------------
        # False-positive spatial distribution
        # -------------------------------------------------

        grid_counts = get_grid_counts(
            fp,
            GRID_SIZE
        )

        class_grids[class_name] += grid_counts

        class_fp_totals[class_name] += fp_count

        # Border and interior FP counts
        border_fp = int(
            np.logical_and(fp, border_mask).sum()
        )

        interior_fp = fp_count - border_fp

        class_border_totals[class_name][
            "border_fp"
        ] += border_fp

        class_border_totals[class_name][
            "interior_fp"
        ] += interior_fp

        border_percentage = (
            100 * border_fp / fp_count
            if fp_count > 0
            else np.nan
        )

        # FP centroid
        centroid_y, centroid_x = get_centroid(fp)

        # Quadrant distribution
        quadrants = get_quadrant_counts(fp)

        # -------------------------------------------------
        # Per-image report
        # -------------------------------------------------

        per_image_rows.append({
            "image_id": image_id,
            "class": class_name,
            "threshold": threshold,

            "gt_pixels": gt_count,
            "predicted_pixels": pred_count,

            "TP_pixels": tp_count,
            "FP_pixels": fp_count,
            "FN_pixels": fn_count,

            "FP_percentage_of_prediction": (
                100 * fp_count / pred_count
                if pred_count > 0
                else np.nan
            ),

            "FP_centroid_y_normalized": centroid_y,
            "FP_centroid_x_normalized": centroid_x,

            "border_FP_pixels": border_fp,
            "interior_FP_pixels": interior_fp,
            "border_FP_percentage": border_percentage,

            **{
                f"FP_{key}": value
                for key, value in quadrants.items()
            },

            "original_height": original_size[0],
            "original_width": original_size[1],
        })

        # -------------------------------------------------
        # Store examples with highest FP count
        # -------------------------------------------------

        if fp_count > 0:

            example_candidates[class_name].append({
                "image_id": image_id,
                "fp_count": fp_count,
                "image": display_image.copy(),
                "gt": gt.copy(),
                "pred": pred.copy(),
                "fp": fp.copy(),
                "fn": fn.copy(),
            })


# =====================================================
# 8. CREATE SPATIAL GRID SUMMARY
# =====================================================

for class_name in CLASS_NAMES:

    grid = class_grids[class_name]

    total_fp = class_fp_totals[class_name]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            count = int(grid[row, col])

            percentage = (
                100 * count / total_fp
                if total_fp > 0
                else np.nan
            )

            grid_summary_rows.append({
                "class": class_name,
                "grid_row": row + 1,
                "grid_column": col + 1,
                "FP_pixels": count,
                "percentage_of_class_FP": percentage,
            })


# =====================================================
# 9. CREATE BORDER SUMMARY
# =====================================================

border_summary_rows = []

for class_name in CLASS_NAMES:

    border_fp = class_border_totals[class_name][
        "border_fp"
    ]

    interior_fp = class_border_totals[class_name][
        "interior_fp"
    ]

    total_fp = border_fp + interior_fp

    border_summary_rows.append({
        "class": class_name,
        "border_width_pixels": BORDER_WIDTH,
        "border_FP_pixels": border_fp,
        "interior_FP_pixels": interior_fp,
        "total_FP_pixels": total_fp,

        "border_FP_percentage": (
            100 * border_fp / total_fp
            if total_fp > 0
            else np.nan
        ),

        "interior_FP_percentage": (
            100 * interior_fp / total_fp
            if total_fp > 0
            else np.nan
        ),
    })


# =====================================================
# 10. SAVE VISUALIZATIONS
# =====================================================

print("\nGenerating visualizations...")

for class_name in CLASS_NAMES:

    # Aggregate spatial heatmap
    save_spatial_heatmap(
        class_grids[class_name],
        class_name
    )

    # Sort examples by FP pixel count.
    candidates = sorted(
        example_candidates[class_name],
        key=lambda item: item["fp_count"],
        reverse=True
    )

    selected_examples = candidates[:TOP_K_EXAMPLES]

    for example in selected_examples:

        save_error_visualization(
            image=example["image"],
            gt=example["gt"],
            pred=example["pred"],
            fp=example["fp"],
            fn=example["fn"],
            class_name=class_name,
            image_id=example["image_id"],
        )


# =====================================================
# 11. SAVE CSV REPORTS
# =====================================================

per_image_df = pd.DataFrame(per_image_rows)

grid_df = pd.DataFrame(grid_summary_rows)

border_df = pd.DataFrame(border_summary_rows)

per_image_path = (
    OUTPUT_DIR / "per_image_spatial_errors.csv"
)

grid_path = (
    OUTPUT_DIR / "spatial_grid_summary.csv"
)

border_path = (
    OUTPUT_DIR / "border_error_summary.csv"
)

per_image_df.to_csv(
    per_image_path,
    index=False
)

grid_df.to_csv(
    grid_path,
    index=False
)

border_df.to_csv(
    border_path,
    index=False
)


# =====================================================
# 12. DISPLAY SUMMARY
# =====================================================

print("\n" + "=" * 85)
print("SPATIAL ERROR ANALYSIS SUMMARY")
print("=" * 85)

print("\nBORDER VS INTERIOR FALSE POSITIVES")

print(
    border_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nSPATIAL GRID DISTRIBUTION")

for class_name in CLASS_NAMES:

    print(f"\n{class_name}")

    print(
        pd.DataFrame(
            class_grids[class_name]
        ).to_string(index=False, header=False)
    )

print("\n" + "=" * 85)
print("SPATIAL ERROR ANALYSIS COMPLETED")
print("=" * 85)

print("\nReports saved:")
print(per_image_path)
print(grid_path)
print(border_path)

print("\nVisualizations saved:")
print(VISUAL_DIR)

print("\nRepresentative examples saved:")
print(EXAMPLE_DIR)

print("\nTotal per-image records:", len(per_image_df))
print("Total grid records:", len(grid_df))
print("Total border records:", len(border_df))
