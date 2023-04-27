import click
import numpy as np
import pandas as pd
from pathlib import Path
import random
import tqdm
from PIL import Image
import shutil as sh
import json
from sklearn.model_selection import train_test_split
import sys

# if 'kaggle' in os.getcwd():
# 	print("CHECK______")
# 	sys.path.append('/kaggle/working/OilTankVolumeEstimation/')
# 	print(sys.path)

import albumentations as A
import cv2

from utils import conv_bbox, mkdir, config_parse, dataset_download


def get_annotations(json_labels:list, dataset_path, train_path, test_path) -> pd.DataFrame:
	label_to_num = {'Tank': 0, 'Tank Cluster': 1, 'Floating Head Tank': 2}
	df = []
	for i, obj in tqdm.tqdm(enumerate(json_labels), desc='Creating train/test paths...'):
		file_name = obj['file_name']
		if i%10 == 0:
			sh.copy(dataset_path / 'image_patches' / file_name, test_path)
		elif obj['label'] != 'Skip':
			sh.copy(dataset_path / 'image_patches' / file_name, train_path)
			for label in obj['label'].keys():
				class_name = label
				class_id = label_to_num[class_name]
				for bbox_coord in obj['label'][class_name]:
					y_min, x_min, y_max, x_max = conv_bbox(bbox_coord['geometry'])
					width = x_max - x_min
					height = y_max - y_min
					df.append([file_name.split('.')[0], class_name, class_id, [x_min, y_min, width, height]])
	df = pd.DataFrame(df, columns=['image_name', 'class_name', 'class_id', 'bbox'])
	annotations = df.groupby('image_name')['bbox'].apply(list).reset_index(name='bboxes')
	annotations['classes'] = df.groupby('image_name')['class_id'].apply(list).reset_index(name='classes')['classes']
	return annotations


def augmentation(image_features, annotations, dataset_path, train_path):
	transforms = A.Compose([
		A.OneOf(
			[A.Flip(),
			 A.Rotate(),
			 A.HorizontalFlip()], p=1)
	], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

	image_path = str(dataset_path / 'image_patches' / (image_features['image_name'] + '.jpg'))
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	bboxes = image_features['bboxes']
	classes = image_features['classes']
	transformer = transforms(image=image, bboxes=bboxes, class_labels=classes)
	transformed_image = transformer['image']
	transformed_bboxes = transformer['bboxes']
	transformed_class_labels = transformer['class_labels']
	result = Image.fromarray(transformed_image, mode='RGB')
	result.save(train_path / (image_features['image_name'] + '_aug.jpg'))
	annotations.loc[len(annotations.index)] = [image_features['image_name'] + '_aug',
											   transformed_bboxes, transformed_class_labels]


def processing(data, mode, working_path, yolo_dataset_path, train_path):
	OUTPUT_IMAGE_FILE = Path(working_path / 'datasets' / yolo_dataset_path / 'images' / mode)
	OUTPUT_LABEL_FILE = Path(working_path / 'datasets' / yolo_dataset_path / 'labels' / mode)

	for _, row in tqdm.tqdm(data.iterrows(), desc=f'Creating {mode} yolo data...'):
		image_name = row['image_name']
		bounding_boxes = row['bboxes']
		classes = row['classes']
		yolo_data = []
		for bbox, Class in zip(bounding_boxes, classes):
			x = bbox[0]
			y = bbox[1]
			w = bbox[2]
			h = bbox[3]
			x_center = x + w / 2
			y_center = y + h / 2

			x_center /= 512
			y_center /= 512
			w /= 512
			h /= 512
			yolo_data.append([Class, x_center, y_center, w, h])
		np.savetxt(OUTPUT_LABEL_FILE / (image_name + '.txt'), yolo_data, fmt='%f')
		sh.copy(train_path / (image_name + '.jpg'), OUTPUT_IMAGE_FILE)


config = config_parse('config.yaml')
@click.command()
@click.option('--dataset_path', type=str, default=config['build']['raw_data_file'], help='path of raw dataset')
@click.option('--path_to', type=str, default=config['build']['path_to'], help='path of processing dataset')
@click.option('--random_state', type=int, default=config['build']['random_state'])
@click.option('--test_size', type=float, default=config['build']['test_size'])
@click.option('--augment', type=bool, default=config['build']['augment'])
def create_yolo_dataset(dataset_path, path_to, random_state, test_size, augment):
	WORKING_PATH = Path('../data')
	DATASET_PATH = Path(WORKING_PATH / dataset_path)
	if dataset_path == 'oiltank':
		dataset_download(DATASET_PATH)
	train_path = Path(WORKING_PATH / 'train')
	test_path = Path(WORKING_PATH / 'test')
	mkdir(test_path)
	mkdir(train_path)

	json_labels = json.load(open(DATASET_PATH / 'labels.json'))
	annotations = get_annotations(json_labels, DATASET_PATH, train_path, test_path)
	if augment:
		for _, row in tqdm.tqdm(annotations.iterrows(), desc='Image_augmentating...'):
			augmentation(row, annotations, DATASET_PATH, train_path)
	train, val = train_test_split(annotations, test_size=test_size, random_state=random_state, shuffle=True)

	mkdir(WORKING_PATH / 'datasets')
	mkdir(WORKING_PATH / 'datasets' / path_to)
	mkdir(WORKING_PATH / 'datasets' / path_to / 'images')
	mkdir(WORKING_PATH / 'datasets' / path_to / 'labels')
	mkdir(WORKING_PATH / 'datasets' / path_to / 'images' / 'train')
	mkdir(WORKING_PATH / 'datasets' / path_to / 'images'/ 'validation')
	mkdir(WORKING_PATH / 'datasets' / path_to / 'labels' / 'train')
	mkdir(WORKING_PATH / 'datasets' / path_to / 'labels'/ 'validation')

	processing(train, 'train', WORKING_PATH, path_to, train_path)
	processing(val, 'validation', WORKING_PATH, path_to, train_path)


if __name__ == '__main__':
	create_yolo_dataset()


