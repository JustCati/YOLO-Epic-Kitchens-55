import os
import ast
import pandas as pd
from PIL import Image




def compute_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def process_yolo_labels(folder_path, iou_threshold=0.85):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            bboxes = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                bboxes.append((class_id, x_center, y_center, width, height))

            to_remove = set()
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    box1, box2 = bboxes[i], bboxes[j]
                    if box1[0] == box2[0]:  # Same class
                        iou = compute_iou(box1[1:], box2[1:])
                        if iou >= iou_threshold:
                            area1 = box1[3] * box1[4]
                            area2 = box2[3] * box2[4]
                            if area1 < area2:
                                to_remove.add(i)
                            else:
                                to_remove.add(j)

            filtered_bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in to_remove]
            with open(file_path, 'w') as file:
                for bbox in filtered_bboxes:
                    file.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")



def convert_bbox(size, box):
        width, height = size
        y_min, x_min, bbox_height, bbox_width = box
        x_center = (x_min + (x_min + bbox_width)) / 2.0 / width
        y_center = (y_min + (y_min + bbox_height)) / 2.0 / height
        bbox_width /= width
        bbox_height /= height
        return x_center, y_center, bbox_width, bbox_height



def process_data(subset_data, image_dest, label_dest, source_folder, label_map):
        for _, row in subset_data.iterrows():
            participant_id = row["participant_id"]
            video_id = row["video_id"]
            frame_number = row["frame"]
            class_id = label_map[row["noun_class"]]

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



def convert(root_folder, source_folder, yolo_base):
    data = pd.read_csv(os.path.join(root_folder, "EPIC_train_object_labels.csv"))

    yolo_images_train = os.path.join(yolo_base, "images/train")
    yolo_images_test = os.path.join(yolo_base, "images/val")
    yolo_labels_train = os.path.join(yolo_base, "labels/train")
    yolo_labels_test = os.path.join(yolo_base, "labels/val")

    for path in [yolo_images_train, yolo_images_test, yolo_labels_train, yolo_labels_test]:
        os.makedirs(path, exist_ok=True)

    ids = data["video_id"].unique()
    train_id = ids[:int(len(ids) * 0.8)]
    test_id = ids[int(len(ids) * 0.8):]

    train_data = data[data["video_id"].isin(train_id)]
    test_data = data[data["video_id"].isin(test_id)]

    classes = data["noun_class"].unique()
    label_map = {c: i for i, c in enumerate(classes)}

    process_data(train_data, yolo_images_train, yolo_labels_train, source_folder, label_map)
    process_data(test_data, yolo_images_test, yolo_labels_test, source_folder, label_map)

    process_yolo_labels(yolo_labels_train)
    process_yolo_labels(yolo_labels_test)

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
