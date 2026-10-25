
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

MANIFEST_PATH = DATA / "processed" / "validated_manifest.csv"
LABELS_PATH = DATA / "annotations" / "image_labels_train.csv"
BOXES_PATH = DATA / "annotations" / "annotations_train.csv"

st.set_page_config(
    page_title="VinDr-CXR Viewer",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 VinDr-CXR Image Annotation Viewer")
st.caption("Local research prototype | Validated DICOM subset")

# Load project data
@st.cache_data
def load_data():
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"image_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"image_id": str})
    boxes = pd.read_csv(BOXES_PATH, dtype={"image_id": str})
    return manifest, labels, boxes

manifest, labels, boxes = load_data()

# Image selection
image_ids = sorted(manifest["image_id"].dropna().unique().tolist())

selected_id = st.selectbox(
    "Select an image ID",
    image_ids,
    format_func=lambda x: x,
)

row = manifest.loc[manifest["image_id"] == selected_id].iloc[0]
dicom_path = ROOT / row["dicom_path"]

# Match annotations to selected image
image_labels = labels[labels["image_id"] == selected_id].copy()
image_boxes = boxes[boxes["image_id"] == selected_id].copy()


if not image_boxes.empty and "rad_id" in image_boxes.columns:
    radiologists = ["All radiologists"] + sorted(
        image_boxes["rad_id"].dropna().unique().tolist()
    )
    selected_reader = st.selectbox(
        "Show annotations from",
        radiologists,
    )

    if selected_reader != "All radiologists":
        image_boxes = image_boxes[
            image_boxes["rad_id"] == selected_reader
        ]


box_cols = ["x_min", "y_min", "x_max", "y_max"]
image_boxes = image_boxes.dropna(subset=box_cols)

# Display image
try:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array.astype(np.float32)

    # Apply modality rescale where present
    pixels = (
        pixels * float(getattr(ds, "RescaleSlope", 1))
        + float(getattr(ds, "RescaleIntercept", 0))
    )

    # Display normalization
    low, high = np.min(pixels), np.max(pixels)
    if high > low:
        pixels = (pixels - low) / (high - low)
    else:
        pixels = np.zeros_like(pixels)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1 - pixels

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Chest X-ray")

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(pixels, cmap="gray", origin="upper")

        for _, box in image_boxes.iterrows():
            x1 = float(box["x_min"])
            y1 = float(box["y_min"])
            x2 = float(box["x_max"])
            y2 = float(box["y_max"])

            # CSV coordinates are assumed to be pixel coordinates.
            # Confirm alignment against sample images before research use.
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )

            class_name = str(box.get("class_name", "Finding"))
            ax.text(
                x1,
                max(0, y1 - 6),
                class_name,
                color="yellow",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            )

        ax.set_xlim(0, pixels.shape[1])
        ax.set_ylim(pixels.shape[0], 0)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Image information")
        st.write("**Image ID:**", selected_id)
        st.write("**Dimensions:**", f"{ds.Rows} × {ds.Columns}")
        st.write("**Pixel type:**", str(ds.pixel_array.dtype))
        st.write("**Bounding-box rows:**", len(image_boxes))

        if image_boxes.empty:
            st.info("No bounding boxes are available for this image.")
        else:
            st.subheader("Bounding-box annotations")
            st.dataframe(
                image_boxes[
                    ["class_name", "x_min", "y_min", "x_max", "y_max"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Classification labels")
        if image_labels.empty:
            st.warning("No classification-label rows found.")
        else:
            st.dataframe(
                image_labels,
                use_container_width=True,
                hide_index=True,
            )

except Exception as exc:
    st.error(f"Could not display this image: {exc}")
    st.caption(f"File path: {dicom_path}")
