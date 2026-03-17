# 🚨 RAD: Real-Time Accident Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)](https://opencv.org/)

A lightweight, real-time computer vision system designed to detect vehicular accidents from live video streams and CCTV footage. 

Built with edge deployment in mind, this project utilizes a **MobileNetV2-based Convolutional Neural Network (CNN)** optimized for inference efficiency, making it highly effective without requiring massive computational overhead.

## 🎥 Demonstration
![RAD Demo](demo.gif)

## 🧠 Architecture & Engineering Highlights
This system goes beyond basic frame classification by implementing practical engineering solutions for real-world video analysis:

* **Imbalance Handling:** Utilizes Synthetic Minority Over-sampling Technique (SMOTE) on the feature space to resolve severe class imbalance in the 10,000+ image dataset.
* **Temporal Smoothing (Memory Buffer):** Mitigates "frame-by-frame amnesia" using a custom OpenCV temporal buffer. If a collision is detected, the system sustains the emergency alert for 50 frames to ensure first responders don't miss the event after the vehicles settle.
* **Dynamic Thresholding for Domain Shift:** Specifically calibrated to handle low-light, nighttime CCTV footage by adjusting confidence thresholds, successfully identifying accidents even when visual clarity is degraded.
* **Feature Extraction:** Leverages transfer learning via the MobileNetV2 base model.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras (MobileNetV2)
* **Computer Vision:** OpenCV
* **Data Processing:** NumPy, Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Image Augmentation:** Keras ImageDataGenerator

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/WeirdoWrench/RAD.git](https://github.com/WeirdoWrench/RAD.git)
   cd RAD
   ```
2. **Create a virtual environment:**\
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

1. **Running Live Inference (Pre-Trained Model)**
    To test the model on a local video file or webcam stream:

    ```bash
    python inference.py
    ```

    * Press q to exit the video stream early.

    * To use your webcam instead of a video file, change VIDEO_SOURCE = 0 inside inference.py.

    * Note: The CONFIDENCE_THRESHOLD can be adjusted inside the script depending on the lighting conditions of your video source.

2. **Training the Model (Optional)**
    To train the model from scratch using your own dataset (Requires the Kaggle Car Crash dataset Excel format):

    ```bash
    python train.py
    ```
    * Note: Training applies SMOTE and utilizes ImageDataGenerator. Running this in a Linux/WSL environment with a dedicated GPU is highly recommended for performance.

## 🤝 Let's Connect

Created by @Weirdo_Wrench