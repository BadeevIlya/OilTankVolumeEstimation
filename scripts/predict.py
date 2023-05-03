import PIL
from ultralytics import YOLO
import click
from utils import config_parse, get_log, create_diff_dir
from volume_estimating import MultiTank
from skimage import io


config = config_parse('config.yaml')
@click.command()
@click.option('--model', type=str, default=config['predict']['model_weights'])
@click.option('--img_src', type=str, default=config['predict']['img_src'])
def predict(model, img_src):
	model = YOLO(model)
	img_src = 'assets'
	results = model.predict(img_src, save=True, imgsz=4800, conf=0.7, name='Test_predict')
	log = get_log()
	target_path = create_diff_dir()
	for result in results:
		bbox = []
		result = result.numpy()
		path = result.path.split("\\")[-1]
		for i, cls in enumerate(result.boxes.cls):
			if cls == 2:
				bbox.append(result.boxes.xyxy[i])
		orig_img = io.imread(f'{img_src}/{path}')
		tanks = MultiTank(bbox, orig_img, path, log)
		tanks.plot_volumes(target_path)


if __name__ == '__main__':
	predict()