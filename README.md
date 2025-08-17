

# Real-Time Face Classification

This project uses a ResNet50-based PyTorch model to classify faces detected from a webcam stream in real time. The model distinguishes between three classes: `davido`, `rihanna`, and `ronaldo`.

## Features

- Real-time face detection using OpenCV.
- Classification of detected faces using a pretrained and fine-tuned ResNet50 model.
- Displays bounding boxes and predicted labels with confidence scores on the video stream.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (`opencv-python`)
- Pillow

## Setup

1. **Clone the repository** and navigate to the project folder.

2. **Install dependencies**:
    ```bash
    pip install torch torchvision opencv-python pillow
    ```

3. **Model Checkpoint**:
    - Place your trained model checkpoint at:
      ```
      C:\Users\abdul\OneDrive\Documents\images_class\image_classification_model.pth
      ```
    - Ensure the checkpoint matches the model architecture in `stream.py`.

## Usage

Run the following command to start real-time face classification:

```bash
python stream.py
