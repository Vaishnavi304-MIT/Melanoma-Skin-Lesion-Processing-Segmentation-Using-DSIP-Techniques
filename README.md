# Melanoma Skin Lesion Processing Using Digital Image Techniques

> **A computationally simple, cost-effective, and real-time melanoma segmentation system — built without any machine learning or GPU.**

---

## Institution
**MIT Academy of Engineering, Pune**  
Department of Electronics & Telecommunication Engineering

---

## 👥 Team

| Name | PRN |
|---|---|
| Vaishnavi Shinde | 202402060016 |
| Shreya Kalaskar | 202402060006 |
| Pushkar Yadav | 202402060017 |
| Aayush Katamkar | 202402060015 |

**Guided by:** Dr. Smita Kulkarni

---

## Problem Statement

Melanoma is the most dangerous form of skin cancer, responsible for the majority of skin cancer-related deaths worldwide. Early and accurate detection is critical — when identified at **Stage I**, survival rates exceed **98%**; at **Stage IV**, this drops below **20%**. However, visual inspection of dermoscopic images is a skilled clinical task subject to inter-observer variability, making computer-aided analysis highly valuable.

---

## Objectives

- Enhance dermoscopic skin images using **CLAHE**, brightness enhancement, and gamma correction for better lesion visibility.
- Remove noise, hair artifacts, and unwanted texture variations using **spatial filtering techniques**.
- Accurately segment melanoma lesion regions from surrounding healthy skin using **adaptive threshold segmentation**.
- Improve lesion continuity and remove small noisy regions using **morphological operations** and connected component filtering.
- Develop a **low-cost and interpretable** melanoma detection pipeline that does not require GPU processing or large training datasets.
- Detect melanoma lesion boundaries using **contour detection** and **Canny edge detection** techniques.

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** — image processing and computer vision
- **NumPy** — numerical operations
- **Streamlit** — web application UI
- **Digital Image Processing (DIP)** techniques

---

## 12-Stage Processing Pipeline

<svg width="100%" viewBox="0 0 680 1060" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
        stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <style>
      text { font-family: sans-serif; fill: #1a1a1a; }
      .th  { font-size: 14px; font-weight: 600; }
      .ts  { font-size: 12px; fill: #555; }
      .arr { stroke: #888; stroke-width: 1.5; fill: none; }
      /* Phase colors */
      .teal   { fill: #e1f5ee; stroke: #0f6e56; }
      .purple { fill: #eeedfe; stroke: #534ab7; }
      .coral  { fill: #faece7; stroke: #993c1d; }
      .blue   { fill: #e6f1fb; stroke: #185fa5; }
      .gray   { fill: #f1efe8; stroke: #5f5e5a; }
      .teal text, .purple text, .coral text, .blue text, .gray text { fill: #1a1a1a; }
      .ts.teal   { fill: #0f6e56; }
      .ts.purple { fill: #534ab7; }
    </style>
  </defs>

  <!-- INPUT -->
  <g class="gray"><rect x="240" y="20" width="200" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="42" text-anchor="middle" dominant-baseline="central">Input image</text></g>
  <line x1="340" y1="64" x2="340" y2="82" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 1 -->
  <g class="teal"><rect x="210" y="84" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="104" text-anchor="middle" dominant-baseline="central">Stage 1 — Resize</text>
    <text class="ts" x="340" y="122" text-anchor="middle" dominant-baseline="central">All images → 512 × 512 px</text></g>
  <line x1="340" y1="136" x2="340" y2="154" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 2 -->
  <g class="teal"><rect x="210" y="156" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="176" text-anchor="middle" dominant-baseline="central">Stage 2 — Grayscale</text>
    <text class="ts" x="340" y="194" text-anchor="middle" dominant-baseline="central">Color → intensity map</text></g>
  <line x1="340" y1="208" x2="340" y2="226" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 3 -->
  <g class="teal"><rect x="210" y="228" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="248" text-anchor="middle" dominant-baseline="central">Stage 3 — CLAHE</text>
    <text class="ts" x="340" y="266" text-anchor="middle" dominant-baseline="central">8×8 tiles, clipLimit=3.0</text></g>
  <line x1="340" y1="280" x2="340" y2="298" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 4 -->
  <g class="teal"><rect x="210" y="300" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="320" text-anchor="middle" dominant-baseline="central">Stage 4 — Brightness +35</text>
    <text class="ts" x="340" y="338" text-anchor="middle" dominant-baseline="central">Lesion–skin contrast boost</text></g>
  <line x1="340" y1="352" x2="340" y2="370" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 5 -->
  <g class="teal"><rect x="210" y="372" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="392" text-anchor="middle" dominant-baseline="central">Stage 5 — Gamma γ=1.7</text>
    <text class="ts" x="340" y="410" text-anchor="middle" dominant-baseline="central">Darkens lesion boundary pixels</text></g>
  <line x1="340" y1="424" x2="340" y2="442" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 6 -->
  <g class="purple"><rect x="210" y="444" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="464" text-anchor="middle" dominant-baseline="central">Stage 6 — Median filter</text>
    <text class="ts" x="340" y="482" text-anchor="middle" dominant-baseline="central">5×5 kernel, removes hair/noise</text></g>
  <line x1="340" y1="496" x2="340" y2="514" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 7 — Core innovation -->
  <g class="coral"><rect x="200" y="516" width="280" height="56" rx="8" stroke-width="1.5"/>
    <text class="th" x="340" y="536" text-anchor="middle" dominant-baseline="central">Stage 7 — Adaptive threshold ★</text>
    <text class="ts" x="340" y="556" text-anchor="middle" dominant-baseline="central">T = P − (0.28 × P)  |  core innovation</text></g>
  <line x1="340" y1="572" x2="340" y2="590" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 8 -->
  <g class="purple"><rect x="210" y="592" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="612" text-anchor="middle" dominant-baseline="central">Stage 8 — Gaussian smooth</text>
    <text class="ts" x="340" y="630" text-anchor="middle" dominant-baseline="central">5×5, softens binary mask edges</text></g>
  <line x1="340" y1="644" x2="340" y2="662" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 9 -->
  <g class="purple"><rect x="210" y="664" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="684" text-anchor="middle" dominant-baseline="central">Stage 9 — Morph. closing</text>
    <text class="ts" x="340" y="702" text-anchor="middle" dominant-baseline="central">Dilation then erosion, fills holes</text></g>
  <line x1="340" y1="716" x2="340" y2="734" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 10 -->
  <g class="purple"><rect x="210" y="736" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="756" text-anchor="middle" dominant-baseline="central">Stage 10 — Dilation</text>
    <text class="ts" x="340" y="774" text-anchor="middle" dominant-baseline="central">Expands lesion, connects pixels</text></g>
  <line x1="340" y1="788" x2="340" y2="806" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 11 -->
  <g class="blue"><rect x="210" y="808" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="828" text-anchor="middle" dominant-baseline="central">Stage 11 — Connected comp.</text>
    <text class="ts" x="340" y="846" text-anchor="middle" dominant-baseline="central">Remove blobs &lt; 120 px area</text></g>
  <line x1="340" y1="860" x2="340" y2="878" class="arr" marker-end="url(#arrow)"/>

  <!-- Stage 12 -->
  <g class="blue"><rect x="210" y="880" width="260" height="52" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="900" text-anchor="middle" dominant-baseline="central">Stage 12 — Contour + Canny</text>
    <text class="ts" x="340" y="918" text-anchor="middle" dominant-baseline="central">Edge detection, low=50 high=150</text></g>
  <line x1="340" y1="932" x2="340" y2="950" class="arr" marker-end="url(#arrow)"/>

  <!-- OUTPUT -->
  <g class="gray"><rect x="230" y="952" width="220" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="340" y="974" text-anchor="middle" dominant-baseline="central">Segmented lesion output</text></g>

  <!-- Legend -->
  <g class="teal"><rect x="68" y="1010" width="12" height="12" rx="3"/></g>
  <text class="ts" x="86" y="1021">Enhancement (stages 1–5)</text>
  <g class="purple"><rect x="280" y="1010" width="12" height="12" rx="3"/></g>
  <text class="ts" x="298" y="1021">Filtering &amp; morphology (6–10)</text>
  <g class="blue"><rect x="68" y="1030" width="12" height="12" rx="3"/></g>
  <text class="ts" x="86" y="1041">Detection (11–12)</text>
  <g class="coral"><rect x="280" y="1030" width="12" height="12" rx="3"/></g>
  <text class="ts" x="298" y="1041">Core innovation (stage 7)</text>
</svg>
```
Input Image → Resize → Grayscale → CLAHE → Brightness → Gamma
     → Median Filter → Adaptive Threshold → Gaussian Smooth
          → Morphological Closing → Dilation → Connected Components → Contour + Canny Edges
```

| Stage | Step | Description |
|---|---|---|
| 1 | Input & Resize | Resized to 512×512 pixels for consistency |
| 2 | Grayscale Conversion | Focuses on intensity differences between lesion and skin |
| 3 | CLAHE Enhancement | Contrast Limited Adaptive Histogram Equalization (8×8 tiles, clipLimit=3.0) |
| 4 | Brightness +35 | Adds +35 offset to increase lesion-skin contrast |
| 5 | Gamma Correction (γ=1.7) | Non-linear transform; darkens dark pixels to sharpen lesion boundary |
| 6 | Median Filtering | 5×5 kernel — removes noise, hair artifacts, preserves edges |
| 7 | Adaptive Thresholding | Core innovation — auto-adapts to skin tone via histogram |
| 8 | Gaussian Smoothing | 5×5 kernel — softens rough binary mask edges |
| 9 | Morphological Closing | Dilation then Erosion — fills holes, improves lesion continuity |
| 10 | Morphological Dilation | Expands lesion region, connects nearby disconnected pixels |
| 11 | Connected Components | Removes white blobs with area < 120 pixels (noise elimination) |
| 12 | Contour + Canny Edges | Detects precise lesion boundaries (low=50, high=150) |

---

## Core Innovation — Adaptive Threshold Segmentation (Stage 7)

```
T  =  P  −  ( 0.28 × P )
```

Where:
- **T** = threshold value
- **P** = histogram peak (dominant skin intensity)
- **0.28** = scaling factor

**Pixel Classification Rule:**
```
If I(x,y) < T  →  Melanoma (255)
If I(x,y) ≥ T  →  Skin (0)
```

**Why Adaptive?** T is calculated from each image's own histogram — automatically works on all skin tones.

### Comparison with Other Segmentation Methods

| Method | Training Data | GPU Needed | Speed |
|---|---|---|---|
| **Adaptive Threshold (Ours)** | None | No | Fast |
| K-Means Clustering | None | No | Moderate |
| Deep Learning (U-Net) | Large Dataset | Yes | Slow (training) |

---

## 📐 Extracted Morphological Features

| Feature | Formula | Meaning |
|---|---|---|
| Area | Pixel count inside contour | Size of lesion |
| Perimeter | Arc length of boundary | Boundary length |
| Circularity | 4π × Area ÷ Perimeter² | Shape regularity (1.0 = perfect circle) |
| Threshold Value | T = P − (0.28 × P) | Adaptive threshold used |

---

## Advantages & Limitations

### Advantages
- Adapts automatically to different skin tones
- CLAHE + Gamma enhance low-contrast images
- Median filter preserves lesion boundaries while removing noise
- No training data or GPU required
- Computationally fast — suitable for real-time use
- Fully interpretable — every stage is visible
- Accurate boundaries for clear dermoscopic images

### Limitations
- Fails on very low-contrast melanoma lesions
- Sensitive to heavy hair artifacts and shadows
- Less accurate than deep learning (CNN/U-Net)
- Center-based selection fails if lesion is off-center
- Threshold depends on histogram — breaks on heavy noise
- Cannot handle multiple overlapping lesions
- Reduced accuracy for complex pigmentation

**Best for:** Clear dermoscopy images • Central lesions • Dark pigmentation • Moderate-to-high contrast • Uniform illumination

---

## Streamlit Application

### Main Panel
- Title: Melanoma Skin Lesion Image Processing System
- File uploader (JPG / JPEG / PNG)
- 12 output stages displayed with headers
- Histogram comparison (Original vs Enhanced)
- Final contour image with 'Melanoma' label
- Final segmented lesion (color on black background)

### Sidebar
- Feature Metrics: Area (px²), Perimeter (px), Circularity, Threshold value
- Feature DataFrame
- Pipeline block diagram

---

## Project Structure

```
melanoma-dsip/
├── app.py                  # Streamlit application
├── pipeline.py             # 12-stage image processing pipeline
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── sample_images/          # Test dermoscopic images
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/melanoma-dsip.git
cd melanoma-dsip

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Requirements

```
opencv-python
numpy
streamlit
Pillow
```

---

## Conclusion

This project presents a **computationally simple, cost-effective, and real-time** melanoma segmentation system built entirely without machine learning or GPU.

- CLAHE + Gamma + Brightness enhance lesion visibility in low-contrast images
- Adaptive threshold auto-adapts to skin tone — no fixed parameters needed
- Morphological + Component filtering produce clean lesion masks
- Contour + Canny detect precise lesion boundaries
- Extracts Area, Perimeter, Circularity for clinical morphological analysis

---

## Authors

**Vaishnavi Shinde** | **Shreya Kalaskar** | **Pushkar Yadav** | **Aayush Katamkar**  
MIT Academy of Engineering, Electronics & Telecommunication Engineering  
> DSIP Project — Melanoma Skin Lesion Detection System

---
