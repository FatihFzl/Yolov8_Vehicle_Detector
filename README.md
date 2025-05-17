# Vehicle Detection and Tracking System

A comprehensive computer vision project that detects, tracks, and counts vehicles in traffic videos using YOLOv8 and DeepSORT algorithms. This system provides real-time analysis of traffic flow and density.

## Overview

This project implements a complete solution for traffic monitoring and analysis. It can process video footage from traffic cameras to identify different types of vehicles, track their movement, and generate statistics about traffic flow. The system provides a professional visualization interface that displays detection results and traffic metrics in real-time.

**Authors:** Fatih Burak Fazlioglu , Arif Mert Dinc
**Institution:** Bahcesehir University  
**Date:** May 14, 2025

## Features

- **Advanced Vehicle Detection**: Uses YOLOv8, a state-of-the-art object detection model, to identify vehicles with high accuracy
- **Multi-class Vehicle Classification**: Categorizes vehicles into different types (car, truck, van, bus, motorcycle, bicycle)
- **Real-time Tracking**: Implements DeepSORT algorithm to maintain consistent vehicle identity across video frames
- **Traffic Density Analysis**: Calculates and displays current traffic density (LOW/MEDIUM/HIGH) based on vehicle flow
- **Vehicle Counting**: Tracks the number of vehicles by type passing through the monitored area
- **Statistical Visualization**: Displays vehicle counts, percentages, and traffic metrics in an intuitive dashboard
- **Performance Optimization**: Includes frame resizing and processing optimizations for better real-time performance
- **Custom Model Training**: Includes scripts for training and fine-tuning custom YOLOv8 models on vehicle datasets

## Project Structure

```
vehicle-detection-tracking/
├── DATASET/                        # Training and validation dataset
│   ├── train/                      # Training images and labels
│   ├── valid/                      # Validation images and labels
│   ├── data.yaml                   # Dataset configuration
│   └── README files                # Dataset documentation
├── vehicle_detection/              # Model and output directory
│   ├── weights/                    # Trained model weights
│   ├── training_run/               # Training logs and checkpoints
│   └── class_map.yaml              # Class mapping configuration
├── vehicle_detection_tracking.py   # Main detection and tracking script
├── train_yolov8.py                 # Model training script
├── resume_training.py              # Script to resume interrupted training
├── yolov8n.pt                      # Pre-trained YOLOv8 nano model
├── Trafic_Video.mp4                # Sample video for testing
└── requirements.txt                # Required dependencies
```

## Requirements

- Python 3.7+
- OpenCV (for image processing)
- PyYAML (for configuration files)
- NumPy (for numerical operations)
- Ultralytics YOLO (for object detection)
- DeepSORT (for object tracking)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/candeniz-bek/vehicle-detection-tracking.git
cd vehicle-detection-tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download or place your YOLO weights file in the `vehicle_detection/weights/` directory. The default model expected is `best.pt`.

## Scripts and Their Functions

### 1. Main Vehicle Detection and Tracking Script
**File:** `vehicle_detection_tracking.py`

This is the primary script for running the vehicle detection and tracking system. It processes video input to detect, classify, track, and count vehicles in real-time.

**Features:**
- Detects vehicles using a pre-trained YOLOv8 model
- Tracks vehicles across frames using DeepSORT
- Calculates traffic density and statistics
- Provides visual display with bounding boxes and information panel
- Records detection results to output video file

**Usage:**
```bash
# Basic usage with default video
python vehicle_detection_tracking.py

# Specify a custom video file
python vehicle_detection_tracking.py --video path/to/your/video.mp4

# Specify output location for processed video
python vehicle_detection_tracking.py --video path/to/your/video.mp4 --output results.avi
```

**Parameters:**
- `--video`: Path to input video file (default: traffic_video.mp4)
- `--output`: Path for saving processed video with detections and visualizations (optional)

### 2. Model Training Script
**File:** `train_yolov8.py`

This script handles the training of a custom YOLOv8 model on the vehicle dataset. It configures the training parameters, trains the model, and saves the results.

**Features:**
- Loads and configures the dataset from DATASET/data.yaml
- Initializes a pre-trained YOLOv8 model for fine-tuning
- Trains the model with specified hyperparameters
- Saves the best model weights and class mapping
- Handles class compatibility and mapping

**Usage:**
```bash
python train_yolov8.py
```

**Output:**
- Trained model weights in `vehicle_detection/training_run/weights/`
- Best model copied to `vehicle_detection/weights/best.pt`
- Class mapping file created at `vehicle_detection/class_map.yaml`

### 3. Resume Training Script
**File:** `resume_training.py`

This script allows you to resume training from the last checkpoint if the training process was interrupted.

**Features:**
- Loads the last saved model checkpoint
- Continues training from where it left off
- Uses the same configuration as the original training
- Updates the best model and class mapping after completion

**Usage:**
```bash
python resume_training.py
```

**Prerequisites:**
- Previous training run must exist in `vehicle_detection/training_run/`
- Last checkpoint (`last.pt`) must be available

## Dataset Structure

The project uses a custom vehicle detection dataset located in the `DATASET` directory:

- **Source:** UA-DETRAC Dataset (modified version with 10K images)
- **Classes:** bus, car, truck, van
- **Structure:**
  - `train/images/`: Training images
  - `train/labels/`: Training annotations in YOLO format
  - `valid/images/`: Validation images
  - `valid/labels/`: Validation annotations in YOLO format
  - `data.yaml`: Dataset configuration file

The `data.yaml` file configures the dataset paths and class names for the training process.

## Detection Process

The vehicle detection and tracking system works in the following steps:

1. **Frame Acquisition**: Read frames from the input video
2. **Preprocessing**: Resize frames to improve processing speed
3. **Detection**: Apply YOLOv8 model to detect vehicles in each frame
4. **Tracking**: Use DeepSORT to track vehicles across frames and maintain identity
5. **Analysis**: Calculate movement, count vehicles, and determine traffic density
6. **Visualization**: Draw bounding boxes, labels, and information panel
7. **Output**: Display results and save to output video if requested

## Customizing the System

### Using a Custom Model

1. Train your own model using `train_yolov8.py` or provide a pre-trained YOLOv8 model
2. Place the model weights file in `vehicle_detection/weights/best.pt`
3. Update `vehicle_detection/class_map.yaml` to match your model's classes

### Adjusting Detection Parameters

You can modify the following parameters in `vehicle_detection_tracking.py`:

- `detection_threshold`: Minimum confidence score to accept a detection (default: 0.45)
- `motion_threshold`: Pixel distance threshold to consider a vehicle as moving (default: 15)
- DeepSORT tracker parameters: `max_age`, `n_init`, `max_iou_distance`, `max_cosine_distance`

### Modifying Training Parameters

To adjust training configuration, modify the following in `train_yolov8.py`:

- `epochs`: Number of training epochs
- `imgsz`: Input image size
- `batch`: Batch size
- `patience`: Early stopping patience
- Other hyperparameters as needed

## Performance Optimization

For better performance on different hardware:

- Adjust the frame processing rate by changing the frame skip parameter in `vehicle_detection_tracking.py`
- Resize input frames to a smaller size by modifying the `target_width` parameter in `resize_frame()`
- Use a smaller YOLOv8 model (nano or small) for faster inference on less powerful hardware

## Troubleshooting

- **Video Loading Issues**: Ensure your video file format is supported by OpenCV
- **CUDA Errors**: If using GPU, check CUDA/cuDNN compatibility with your PyTorch version
- **Memory Errors**: Reduce batch size during training or input resolution during inference
- **Detection Quality**: Adjust the detection threshold to balance between precision and recall
- **Tracking Stability**: Tune the DeepSORT parameters to improve tracking in challenging scenes

## Understanding the Output

While the program runs, you will see:

1. A window showing the processed video with:
   - Bounding boxes around detected vehicles
   - Vehicle class labels with confidence scores
   - Information panel with statistics and counts
   
2. Terminal output showing:
   - Processing information (FPS, resolution)
   - Summary statistics after completion

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- Deep SORT for the multi-object tracking algorithm
- OpenCV for the computer vision utilities
- UA-DETRAC dataset for providing vehicle detection data

