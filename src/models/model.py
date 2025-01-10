import os
import multiprocessing
from ultralytics import YOLOv10


def getYOLO(checkpoint_path: str, device: str = 'cpu') -> YOLOv10:
    if not os.path.exists(checkpoint_path):
        raise ValueError('Checkpoint path does not exist')
    model = YOLOv10(checkpoint_path)
    model.to(device)
    return model


def train(model: YOLOv10, 
          yaml_file: str,
          epochs: int,
          patience: int,
          batch_size: int,
          model_path: str,
          folder_name: str,
          resume: bool = False,
          device = 'cuda'):
    cpu_workers = multiprocessing.cpu_count()
    results = model.train(data=yaml_file, 
                          batch=batch_size,
                          imgsz=1024,
                          epochs=epochs if patience == 0 else 10000,
                          verbose=True,
                          patience=patience,
                          workers=cpu_workers,
                          save_period=5,
                          device=device,
                          resume=resume,
                          val=True,
                          plots=True,
                          project = model_path,
                          name = folder_name)
    return results