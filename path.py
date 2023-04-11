from pathlib import Path
import os

DATASET_PATH = Path(input("Enter dataset path:"))
WORKING_PATH = Path(os.getcwd())
test_path = Path(WORKING_PATH / 'test')
train_path = Path(WORKING_PATH / 'train')
YOLO_PATH = Path(input("Enter yolo path"))
