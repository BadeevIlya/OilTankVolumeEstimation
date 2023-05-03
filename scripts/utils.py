import numpy as np
import os
import yaml
from zipfile import ZipFile
import gdown
import logging


def create_diff_dir():
	#mask = {"log": "prediction", "volume": "volume_prediction", "contours": "contours"}
	target = "volume_prediction"
	const_len = len(target)
	i = 0
	while os.path.exists(f'runs/detect/{target}'):
		i += 1
		target = f'{target[:const_len]}{i}'
	os.mkdir(f'runs/detect/{target}')
	return target


def create_log_file():
	target = "prediction"
	const_len = len(target)
	i = 0
	while os.path.exists(f'logs/{target}.log'):
		i += 1
		target = f'{target[:const_len]}{i}'
	mkdir('logs')
	return target


def get_log():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler(f"logs/{create_log_file()}.log", mode='w')
	format_log = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
	handler.setFormatter(format_log)
	if (logger.hasHandlers()):
		logger.handlers.clear()
	logger.addHandler(handler)
	return logger


def config_parse(config_file):
	with open(config_file) as f:
		config = yaml.safe_load(f)
	return config


def dataset_download(data_path):
	# url = 'https://drive.google.com/file/d/12PMYpbRdMxufy3FJg50jhsIgCNmtEwqf/view?usp=sharing'
	# output_path = 'oiltank.zip'
	# gdown.download(url, output_path, quiet=False, fuzzy=True)
	with ZipFile('oiltank.zip', mode='r') as archive:
		archive.extractall(data_path)
	os.remove('oiltank.zip')


def conv_bbox(box_dict):
	xs = np.array(list(set([i['x'] for i in box_dict])))
	ys = np.array(list(set([i['y'] for i in box_dict])))
	for i in range(2):
		if xs[i] > 512:
			xs[i] = 512
		if xs[i] < 0:
			xs[i] = 0
		if ys[i] > 512:
			ys[i] = 512
		if ys[i] < 0:
			ys[i] = 0

	x_min = xs.min()
	x_max = xs.max()
	y_min = ys.min()
	y_max = ys.max()

	return y_min, x_min, y_max, x_max


def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)
