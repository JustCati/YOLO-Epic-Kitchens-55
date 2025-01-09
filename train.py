import os
from src.utils.convert_to_yolo import convert



csv_path = os.path.join(os.path.dirname(__file__), "data", "EPIC_train_object_labels.csv")
source_folder = os.path.join(os.path.dirname(__file__), "data", "EPIC-KITCHENS")
yolo_base = os.path.join(os.path.dirname(__file__), "data", "YOLO_dataset")
convert(csv_path, source_folder, yolo_base)
