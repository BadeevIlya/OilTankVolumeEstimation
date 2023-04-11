import os
import ast
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import random
import tqdm
from PIL import Image
import shutil as sh
import json
from string import Template
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sys

if os.getcwd() not in sys.path:
	sys.path.append(os.getcwd())

import albumentations as A
import cv2

from utils.utils import conv_bbox, mkdir
from utils.path import *

def GetAnnotations(json_labels:list) -> pd.DataFrame:
	label_to_num = {'Tank': 0, 'Tank Cluster': 1, 'Floating Head Tank': 2}
	df = []
	for i, obj in tqdm.tqdm(enumerate(json_labels), desc='Creating train/test paths...'):
		file_name = obj['file_name']
		if i%10 == 0:
			sh.copy(DATASET_PATH / 'image_patches' / file_name, test_path)
		elif obj['label'] != 'Skip':
			sh.copy(DATASET_PATH / 'image_patches' / file_name, train_path)
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


def augmentation(image_features, annotations):
	transforms = A.Compose([
		A.OneOf(
			[A.Flip(),
			 A.Rotate(),
			 A.HorizontalFlip()], p=1)
	], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

	image_path = str(DATASET_PATH / 'image_patches' / (image_features['image_name'] + '.jpg'))
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


def processing(data, mode):
	OUTPUT_IMAGE_FILE = Path(YOLO_PATH / 'yolodata' / 'images' / mode)
	OUTPUT_LABEL_FILE = Path(YOLO_PATH / 'yolodata' / 'labels' / mode)

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


def CreateYoloDataset():
	json_labels = json.load(open(DATASET_PATH / 'labels.json'))

	mkdir(test_path)
	mkdir(train_path)

	annotations = GetAnnotations(json_labels)

	for _, row in tqdm.tqdm(annotations.iterrows(), desc='Image_augmentating...'):
		augmentation(row, annotations)

	train, val = train_test_split(annotations, test_size=0.1, random_state = 42)

	mkdir(WORKING_PATH / 'yolodata')
	mkdir(WORKING_PATH / 'yolodata'/ 'images')
	mkdir(WORKING_PATH / 'yolodata'/ 'labels')
	mkdir(WORKING_PATH / 'yolodata'/ 'images' / 'train')
	mkdir(WORKING_PATH / 'yolodata'/ 'images'/ 'validation')
	mkdir(WORKING_PATH / 'yolodata'/ 'labels' / 'train')
	mkdir(WORKING_PATH / 'yolodata'/ 'labels'/ 'validation')

	processing(train, 'train')
	processing(val, 'validation')

CreateYoloDataset()
