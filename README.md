# AI_Archaeological_Mapping
AI-driven system for automated archaeological site monitoring using satellite imagery. A U-Net segmentation model detects structural regions and generates binary masks for erosion analysis. Enables scalable, cost-effective heritage preservation through AI-based image processing and predictive monitoring.
🛰 AI-Driven Archaeological Site Segmentation
📌 Project Overview

This project presents an AI-powered system for detecting archaeological structures from satellite imagery using deep learning. A custom U-Net model performs pixel-level image segmentation to identify structural regions and generate binary masks for analysis.

🎯 Objective

To build a scalable and automated monitoring framework that helps in detecting and analyzing structural patterns in heritage sites using computer vision and AI.

🧠 How It Works

Satellite images are collected and preprocessed.

Masks are automatically generated using adaptive thresholding.

A U-Net segmentation model is trained on image–mask pairs.

The trained model predicts structural regions from new images.

Binary masks are generated for visualization and analysis.

⚙️ Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

📊 Model Details

Architecture: U-Net (Encoder–Decoder CNN)

Input Size: 128x128

Loss Function: Binary Crossentropy

Optimizer: Adam

Output: Pixel-wise probability map

🚀 Features

✔ Automatic mask generation
✔ Deep learning-based segmentation
✔ Model training and prediction pipeline
✔ Binary mask visualization

🔮 Future Improvements

IoU and Dice score evaluation

Time-based erosion comparison

Interactive dashboard integration

Drone-based real-time monitoring

🌍 Impact

This solution demonstrates how Artificial Intelligence can support large-scale heritage preservation by enabling automated, cost-effective satellite-based monitoring systems.
