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
<img width="1360" height="2120" alt="skin_lesion_pipeline" src="https://github.com/user-attachments/assets/2f742712-5e7f-4f76-8158-a924b52e69fe" />


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
