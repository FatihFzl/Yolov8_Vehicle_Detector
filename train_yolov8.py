import os
import yaml
import torch
from ultralytics import YOLO

# Ensure the weights directory exists
os.makedirs('vehicle_detection/weights', exist_ok=True)

# Check and potentially modify data.yaml to include all required classes
data_yaml_path = 'DATASET/data.yaml'
required_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van']

# Read the data.yaml file
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Check if all required classes are in the data.yaml
current_classes = data_config.get('names', [])
print(f"Classes in the dataset: {current_classes}")

# Map class names to ensure compatibility with the requested vehicle types
class_map = {}
for i, class_name in enumerate(current_classes):
    class_map[i] = class_name

print(f"Class mapping: {class_map}")
print(torch.cuda.is_available()) #check whether Cuda is avaible or not
# Initialize the model
model = YOLO('yolov8n.pt')  # Load a small pretrained model for CPU efficiency

# Train the model
if __name__ == "__main__":

    results = model.train(
        data=data_yaml_path,             # Path to data YAML file
        epochs=50,                       # Number of training epochs(at least 50)
        imgsz=640,                       # Input image size
        device= 0,                        # Use CUDA for training (use "cpu" if device=0 doesnt work)
        project='vehicle_detection',     # Project name
        name='training_run',             # Run name
        exist_ok=True,                   # Overwrite existing run
        patience=10,                     # Early stopping patience
        batch=8,                         # Batch size (adjust based on CPU memory)
        save=True,                       # Save the trained model
)

# Copy the best model to the weights directory
best_model_path = os.path.join('vehicle_detection', 'training_run', 'weights', 'best.pt')
if os.path.exists(best_model_path):
    import shutil
    shutil.copy(best_model_path, 'vehicle_detection/weights/best.pt')
    print(f"Best model saved to: vehicle_detection/weights/best.pt")
else:
    print("Training completed but could not find the best model path.")

# Create and save a class mapping file for the detection script
class_mapping = {
    i: name for i, name in enumerate(current_classes)
}

# Save class mapping to file
with open('vehicle_detection/class_map.yaml', 'w') as f:
    yaml.dump(class_mapping, f)
    
print("Class mapping saved to: vehicle_detection/class_map.yaml") 