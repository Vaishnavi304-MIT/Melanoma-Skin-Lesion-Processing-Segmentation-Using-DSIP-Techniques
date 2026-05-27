import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import pandas as pd
import io

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Melanoma Skin Lesion Detection",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FF;
}

h1 {
    color: #FF4B4B;
    text-align: center;
    font-size: 46px;
}

.stFileUploader {
    background-color: #E3F2FD;
    padding: 10px;
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: #DCEBFF;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PIPELINE BLOCK DIAGRAM
# =========================================================

def draw_block_diagram():

    steps = [

        ("Input Image", "#2ECC71"),
        ("Resize 512x512", "#3498DB"),
        ("Grayscale Conversion", "#9B59B6"),
        ("CLAHE Enhancement", "#E67E22"),
        ("Brightness Enhancement", "#F39C12"),
        ("Gamma Correction", "#D35400"),
        ("Median Filtering", "#1ABC9C"),
        ("Adaptive Threshold", "#8E44AD"),
        ("Gaussian Smoothing", "#16A085"),
        ("Morphological Closing", "#C0392B"),
        ("Morphological Dilation", "#7F8C8D"),
        ("Connected Component Filtering", "#2980B9"),
        ("Contour Detection", "#27AE60"),
        ("Canny Edge Detection", "#E74C3C"),
        ("Final Segmented Lesion", "#2C3E50")
    ]

    n = len(steps)

    fig, ax = plt.subplots(
        figsize=(4, n * 0.85)
    )

    fig.patch.set_facecolor("#DCEBFF")

    ax.set_xlim(0, 10)

    ax.set_ylim(0, n * 1.2)

    ax.axis("off")

    for i, (text, color) in enumerate(steps):

        y = n * 1.1 - i * 1.1

        rect = FancyBboxPatch(

            (1, y),
            8,
            0.6,

            boxstyle="round,pad=0.05",

            facecolor=color,
            edgecolor="white",
            linewidth=2
        )

        ax.add_patch(rect)

        ax.text(

            5,
            y + 0.3,
            text,

            ha="center",
            va="center",

            color="white",
            fontsize=8,
            fontweight="bold"
        )

        # =================================================
        # DOWNWARD ARROWS
        # =================================================

        if i < n - 1:

            ax.annotate(

                "",

                xy=(5, y - 0.5),      # Arrow head
                xytext=(5, y - 0.1),  # Arrow start

                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2,
                    color="black"
                )
            )

    return fig

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Feature Extraction")

feature_container = st.sidebar.container()

st.sidebar.markdown("---")

st.sidebar.title("Pipeline Diagram")

fig = draw_block_diagram()

buf = io.BytesIO()

fig.savefig(
    buf,
    format="png",
    bbox_inches="tight",
    dpi=120
)

buf.seek(0)

# UPDATED STREAMLIT CODE
st.sidebar.image(
    buf,
    width="stretch"
)

plt.close(fig)

# =========================================================
# TITLE
# =========================================================

st.title("Melanoma Skin Lesion Image Processing System")

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Melanoma Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # INPUT IMAGE
    # =====================================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_np = np.array(image)

    resized = cv2.resize(
        image_np,
        (512, 512)
    )

    # =====================================================
    # STAGE 1 : GRAYSCALE
    # =====================================================

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY
    )

    # =====================================================
    # STAGE 2 : CLAHE
    # =====================================================

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    clahe_output = clahe.apply(
        gray
    )

    # =====================================================
    # STAGE 3 : BRIGHTNESS ENHANCEMENT
    # =====================================================

    brightness = 35

    bright_image = cv2.convertScaleAbs(
        clahe_output,
        alpha=1.0,
        beta=brightness
    )

    # =====================================================
    # HISTOGRAMS
    # =====================================================

    hist_original = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_enhanced = cv2.calcHist(
        [bright_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    # =====================================================
    # STAGE 4 : GAMMA CORRECTION
    # =====================================================

    gamma = 1.7

    normalized = bright_image / 255.0

    gamma_corrected = np.power(
        normalized,
        gamma
    )

    gamma_corrected = np.uint8(
        gamma_corrected * 255
    )

    # =====================================================
    # STAGE 5 : MEDIAN FILTERING
    # =====================================================

    median = cv2.medianBlur(
        gamma_corrected,
        5
    )

    # =====================================================
    # STAGE 6 : ADAPTIVE THRESHOLD
    # =====================================================

    hist = cv2.calcHist(
        [median],
        [0],
        None,
        [256],
        [0, 256]
    )

    skin_peak = np.argmax(
        hist
    )

    dark_threshold = int(
        skin_peak - (0.28 * skin_peak)
    )

    dark_threshold = max(
        dark_threshold,
        45
    )

    adaptive_mask = np.where(
        median < dark_threshold,
        255,
        0
    ).astype(np.uint8)

    # =====================================================
    # STAGE 7 : GAUSSIAN SMOOTHING
    # =====================================================

    smooth_mask = cv2.GaussianBlur(
        adaptive_mask,
        (5, 5),
        0
    )

    _, smooth_mask = cv2.threshold(
        smooth_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # =====================================================
    # STAGE 8 : MORPHOLOGICAL CLOSING
    # =====================================================

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    closing = cv2.morphologyEx(
        smooth_mask,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    # =====================================================
    # STAGE 9 : MORPHOLOGICAL DILATION
    # =====================================================

    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    dilated = cv2.dilate(
        closing,
        dilation_kernel,
        iterations=1
    )

    # =====================================================
    # STAGE 10 : CONNECTED COMPONENT FILTERING
    # =====================================================

    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
        dilated,
        connectivity=8
    )

    filtered_mask = np.zeros_like(
        dilated
    )

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area > 120:

            filtered_mask[
                labels_cc == i
            ] = 255

    # =====================================================
    # STAGE 11 : CONTOUR DETECTION
    # =====================================================

    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_image = resized.copy()

    lesion_mask = np.zeros_like(
        gray
    )

    lesion_data = []

    # =====================================================
    # CENTER CONTOUR SELECTION
    # =====================================================

    best_contour = None

    best_distance = 1e9

    center_x = 256
    center_y = 256

    for cnt in contours:

        area = cv2.contourArea(
            cnt
        )

        if area < 150:
            continue

        M = cv2.moments(
            cnt
        )

        if M["m00"] == 0:
            continue

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
        )

        distance = np.sqrt(
            (cx - center_x) ** 2 +
            (cy - center_y) ** 2
        )

        if distance < best_distance:

            best_distance = distance

            best_contour = cnt

    # =====================================================
    # DRAW FINAL CONTOUR
    # =====================================================

    if best_contour is not None:

        cnt = best_contour

        area = cv2.contourArea(
            cnt
        )

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter**2 + 1e-5
        )

        M = cv2.moments(
            cnt
        )

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
        )

        cv2.drawContours(
            contour_image,
            [cnt],
            -1,
            (0, 255, 0),
            3
        )

        cv2.drawContours(
            lesion_mask,
            [cnt],
            -1,
            255,
            -1
        )

        cv2.putText(
            contour_image,
            "Melanoma",
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        lesion_data.append({

            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2),
            "Circularity": round(circularity, 4),
            "Centroid X": cx,
            "Centroid Y": cy
        })

    # =====================================================
    # STAGE 12 : CANNY EDGE DETECTION
    # =====================================================

    canny_edges = cv2.Canny(
        lesion_mask,
        50,
        150
    )

    # =====================================================
    # FINAL SEGMENTED LESION
    # =====================================================

    segmented_lesion = cv2.bitwise_and(
        resized,
        resized,
        mask=lesion_mask
    )

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    with feature_container:

        if lesion_data:

            df = pd.DataFrame(
                lesion_data
            )

            st.metric(
                "Area",
                f"{df['Area'].sum():.0f}"
            )

            st.metric(
                "Perimeter",
                f"{df['Perimeter'].sum():.0f}"
            )

            st.metric(
                "Threshold",
                dark_threshold
            )

            # UPDATED STREAMLIT CODE
            st.dataframe(
                df,
                width="stretch",
                hide_index=True
            )

        else:

            st.warning(
                "No melanoma detected"
            )

    # =====================================================
    # DISPLAY OUTPUTS
    # =====================================================

    st.header("Stage 1 : Original Image")
    st.image(resized, width=350)

    st.header("Stage 2 : CLAHE Enhancement")
    st.image(clahe_output, width=350)

    st.header("Stage 3 : Brightness Enhancement")
    st.image(bright_image, width=350)

    st.header("Histogram Comparison")

    fig_hist, ax = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    ax[0].plot(hist_original)
    ax[0].set_title("Original Histogram")

    ax[1].plot(hist_enhanced)
    ax[1].set_title("Enhanced Histogram")

    st.pyplot(fig_hist)

    plt.close(fig_hist)

    st.header("Stage 4 : Gamma Correction")
    st.image(gamma_corrected, width=350)

    st.header("Stage 5 : Median Filtering")
    st.image(median, width=350)

    st.header("Stage 6 : Adaptive Threshold")
    st.image(adaptive_mask, width=350)

    st.header("Stage 7 : Gaussian Smoothed Mask")
    st.image(smooth_mask, width=350)

    st.header("Stage 8 : Morphological Closing")
    st.image(closing, width=350)

    st.header("Stage 9 : Morphological Dilation")
    st.image(dilated, width=350)

    st.header("Stage 10 : Connected Component Filtering")
    st.image(filtered_mask, width=350)

    st.header("Stage 11 : Final Contour Detection")
    st.image(contour_image, width=450)

    st.header("Stage 12 : Canny Edge Detection")
    st.image(canny_edges, width=350)

    st.header("Final Segmented Melanoma Lesion")
    st.image(segmented_lesion, width=450)