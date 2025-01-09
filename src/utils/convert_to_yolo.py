import os
import ast
import pandas as pd
from PIL import Image



def convert_bbox(size, box):
        width, height = size
        y_min, x_min, bbox_height, bbox_width = box
        x_center = (x_min + (x_min + bbox_width)) / 2.0 / width
        y_center = (y_min + (y_min + bbox_height)) / 2.0 / height
        bbox_width /= width
        bbox_height /= height
        return x_center, y_center, bbox_width, bbox_height



def process_data(subset_data, image_dest, label_dest, source_folder):
        for _, row in subset_data.iterrows():
            participant_id = row["participant_id"]
            video_id = row["video_id"]
            frame_number = row["frame"]
            class_id = row["noun_class"]

            bbox_list = ast.literal_eval(row["bounding_boxes"])
            if not bbox_list:  
                continue

            source_image = os.path.join(
                source_folder, f"{participant_id}", "object_detection_images", video_id, f"{frame_number:010d}.jpg"
            )
            dest_image = os.path.join(image_dest, f"{video_id}_{frame_number:010d}.jpg")

            if os.path.exists(source_image):
                try:
                    os.symlink(source_image, dest_image)
                except FileExistsError:
                    pass
                with Image.open(source_image) as img:
                    image_size = img.size
            else:
                print(f"Warning: {source_image} does not exist!")
                continue

            label_file = os.path.join(label_dest, f"{video_id}_{frame_number:010d}.txt")
            with open(label_file, "a") as f:
                for bbox in bbox_list:
                    yolo_bbox = convert_bbox(image_size, bbox)
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")



def convert(csv_path, source_folder, yolo_base):
    data = pd.read_csv(csv_path)

    yolo_images_train = os.path.join(yolo_base, "images/train")
    yolo_images_test = os.path.join(yolo_base, "images/val")
    yolo_labels_train = os.path.join(yolo_base, "labels/train")
    yolo_labels_test = os.path.join(yolo_base, "labels/val")

    for path in [yolo_images_train, yolo_images_test, yolo_labels_train, yolo_labels_test]:
        os.makedirs(path, exist_ok=True)

    train_data = data[data["participant_id"] != "P31"]
    test_data = data[data["participant_id"] == "P31"]

    process_data(train_data, yolo_images_train, yolo_labels_train, source_folder)
    process_data(test_data, yolo_images_test, yolo_labels_test, source_folder)

    classes = data["noun_class"].unique()
    label_map = {c: i for i, c in enumerate(classes)}

    noun_classes = {row['noun_class']: row['noun'] for _, row in data.iterrows()}
    noun_classes = {k: v for k, v in sorted(noun_classes.items(), key=lambda item: item[0])}
    noun_classes = {label_map[k]: v for k, v in noun_classes.items()}

    yaml_content = f"""path: {yolo_base}\ntrain: images/train\nval: images/val\n\nnames:\n  {noun_classes}"""
    yaml_content = yaml_content.replace("{", "")
    yaml_content = yaml_content.replace("}", "")
    yaml_content = yaml_content.replace("'", "")
    yaml_content = yaml_content.replace(",", "\n ")

    yaml_path = os.path.join(yolo_base, "dataset.yml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)
    print("Dataset preparation complete.")
    return yaml_path
