import os
import yaml
from ultralytics import YOLO
import shutil

# Paths
data_yaml_path = 'DATASET/data.yaml'
last_model_path = 'vehicle_detection/training_run/weights/last.pt'
best_model_output = 'vehicle_detection/weights/best.pt'
class_map_output = 'vehicle_detection/class_map.yaml'

# Load classes from YAML
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

current_classes = data_config.get('names', [])
print(f"Classes in the dataset: {current_classes}")

# Load last checkpoint model
model = YOLO(last_model_path)

# Resume training
if __name__ == "__main__":
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=6
        ,
        workers=0,  # RAM dostu
        device=0,   # GPU
        resume=True,
        project='vehicle_detection',
        name='training_run',
        exist_ok=True,
        patience=10,
        save=True,
    )

    # Save best model again (if exists)
    best_model_path = os.path.join('vehicle_detection', 'training_run', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, best_model_output)
        print(f"Best model saved to: {best_model_output}")
    else:
        print("Training finished, but best.pt not found.")

    # Save class mapping again
    class_mapping = {i: name for i, name in enumerate(current_classes)}
    with open(class_map_output, 'w') as f:
        yaml.dump(class_mapping, f)
    print(f"Class mapping saved to: {class_map_output}")
