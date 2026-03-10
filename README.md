# AI-Driven Archaeological Site Mapping

## Problem Statement

Archaeologists often rely on manual analysis of satellite and drone imagery to identify ancient ruins and archaeological sites. This process is time-consuming and large geographical regions remain unexplored. Many potential heritage sites are also at risk due to vegetation growth, erosion, and land changes.

## Proposed Solution

This project builds an AI-based system that analyzes satellite imagery to automatically detect important land features and assist archaeologists in identifying potential archaeological regions.

## Project Modules

1. **Dataset Collection**

   * Satellite images and segmentation masks

2. **Data Preprocessing**

   * Dataset organization
   * Image resizing
   * Train/validation/test split

3. **Semantic Segmentation**

   * U-Net deep learning model
   * Pixel-level classification of satellite imagery

4. **Object Detection (Planned)**

   * YOLO-based artifact detection

5. **Erosion Prediction (Planned)**

   * Machine learning model to identify erosion-prone areas

6. **Visualization Dashboard (Planned)**

   * Interactive map interface using Streamlit

## Dataset

The dataset consists of satellite images and corresponding segmentation masks.
Each image has a mask that labels land features such as vegetation, water bodies, and structural regions.

## Technologies Used

* Python
* OpenCV
* PyTorch
* NumPy
* Streamlit

## Project Structure

```
dataset/
scripts/
models/
dashboard/
outputs/
```

## Current Progress

✔ Dataset preparation
✔ Segmentation masks collected
✔ U-Net model implementation
✔ GitHub repository setup

## Future Work

* Train YOLO artifact detection model
* Implement erosion prediction model
* Build interactive dashboard for visualization

## Author

Akash Gowda N C
