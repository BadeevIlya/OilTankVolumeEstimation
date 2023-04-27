from ultralytics import YOLO
import clearml
import click
import sys
import os

from utils import config_parse


config = config_parse('config.yaml')
@click.command()
@click.option('--data', type=str, default=config['train']['data'], help='path of yaml file')
@click.option('--imgsz', type=int, default=config['train']['imgsz'])
@click.option('--batch', type=int, default=config['train']['batch'])
@click.option('--epochs', type=int, default=config['train']['epochs'])
@click.option('--model_weights', type=str, default=config['train']['model_weights'], help='choose pretrained model')
def train(data, imgsz, batch, epochs, model_weights):
	model = YOLO(model_weights)
	results = model.train(
		data= data,
		imgsz=imgsz,
		batch=batch,
		epochs=epochs,
		name='yolov8s_50e_high_resolution_new',
		project='OilTankDetecting')


if __name__ == "__main__":
	train()