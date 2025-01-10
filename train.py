import os
import argparse

from src.utils.convert_to_yolo import convert
from src.models.model import train, getYOLO

import warnings
warnings.filterwarnings("ignore")


def main(args):
    scratch = False
    model_path = args.model_path
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    if model_path == '':
        scratch = True
        model_path = os.path.join(os.path.dirname(__file__), 'ckpts', "YOLO")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dataset_path = args.path
    csv_path = os.path.join(os.path.dirname(dataset_path),"EPIC_train_object_labels.csv")
    yolo_base = os.path.join(os.path.dirname(dataset_path), "YOLO_dataset")

    if not os.path.exists(yolo_base):
        yaml_path = convert(csv_path, dataset_path, yolo_base)
    else:
        yaml_path = os.path.join(yolo_base, "dataset.yml")

    #* Load YOLO model
    device = args.gpu if args.gpu != '' else 'cpu'
    model_file = args.checkpoint if args.checkpoint != '' else 'last.pt'
    yolo_checkpoint_path = os.path.join("pretrained", "yolov10m.pt") if scratch else os.path.join(model_path, "weights", model_file)

    yolo_model = getYOLO(checkpoint_path=yolo_checkpoint_path, device=device)
    print("YOLO model loaded successfully")

    #* Train YOLO model
    epochs = args.epochs
    batch_size = args.batch_size
    train(model=yolo_model, 
        yaml_file=yaml_path,
        batch_size=batch_size,
        epochs=epochs,
        patience=args.patience,
        model_path=model_path if scratch else os.path.dirname(model_path),
        folder_name=os.path.basename(model_path) if not scratch else "YOLOv10",
        resume=not scratch,
        device=device
        )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to data folder', default=os.path.join(os.path.dirname(__file__), "data", "EPIC-KITCHENS"))
    parser.add_argument('--model_path', type=str, default='/', help='Path to model checkpoints folder')
    parser.add_argument('--gpu', type=str, default='', help='Device to train on', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--checkpoint', type=str, default='', help='Name of checkpoint file to resume training')
    args = parser.parse_args()
    main(args)
