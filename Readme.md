# AI-Powered Image Recognition & Object Detection System 🤖👁️

An advanced deep learning system built with Python that combines state-of-the-art computer vision models for image classification and real-time multi-object detection.

## 💡 Project Overview
This project leverages pre-trained Convolutional Neural Networks (CNNs) and fine-tunes them to handle complex visual recognition tasks. It integrates two powerful deep learning architectures: **ResNet-50** for accurate image classification (specifically identifying countries/landmarks from images) and **Faster R-CNN** for real-time bounding-box object detection.

## ⚙️ Key Features
* **Dual-Model Architecture:** Combines classification and object detection pipelines into a unified workflow.
* **Image Classification (ResNet-50):** Utilizes deep residual learning to classify input images and identify specific features or country-related visuals with high precision.
* **Real-Time Object Detection (Faster R-CNN):** Employs Region Proposal Networks (RPN) to accurately detect and locate multiple objects within a single frame in real-time.
* **Pre-trained Weights & Fine-Tuning:** Leverages transfer learning on robust baseline models to achieve optimal performance with specialized datasets.

## 🛠️ Technologies & Libraries Used
* **Language:** Python 3.
* **Deep Learning Frameworks:** PyTorch / TensorFlow (depending on your implementation), TorchVision.
* **Computer Vision Models:** ResNet-50, Faster R-CNN.
* **Data Processing & Utilities:** NumPy, OpenCV, Pandas.

## 🚀 How It Works
1. **Input Processing:** The system takes an image or a video stream as input and preprocesses it (resizing, normalization).
2. **Classification Pipeline:** The image passes through the fine-tuned ResNet-50 network to extract deep spatial features and determine the primary classification category (e.g., country identification).
3. **Detection Pipeline:** Concurrently, Faster R-CNN analyzes the regions of interest (RoIs) to classify and draw bounding boxes around multiple target objects detected in the scene.
4. **Output Generation:** Displays the final annotated image with class labels, confidence scores, and detection bounding boxes.