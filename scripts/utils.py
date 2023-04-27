import numpy as npimport osimport cv2from matplotlib import pyplot as pltimport yamlfrom zipfile import ZipFileimport gdowndef config_parse(config_file):	with open(config_file) as f:		config = yaml.safe_load(f)	return configdef dataset_download(data_path):	# url = 'https://drive.google.com/file/d/12PMYpbRdMxufy3FJg50jhsIgCNmtEwqf/view?usp=sharing'	# output_path = 'oiltank.zip'	# gdown.download(url, output_path, quiet=False, fuzzy=True)	with ZipFile('oiltank.zip', mode='r') as archive:		archive.extractall(data_path)	# os.remove('oiltank.zip')def conv_bbox(box_dict):	xs = np.array(list(set([i['x'] for i in box_dict])))	ys = np.array(list(set([i['y'] for i in box_dict])))	for i in range(2):		if xs[i] > 512:			xs[i] = 512		if xs[i] < 0:			xs[i] = 0		if ys[i] > 512:			ys[i] = 512		if ys[i] < 0:			ys[i] = 0	x_min = xs.min()	x_max = xs.max()	y_min = ys.min()	y_max = ys.max()	return y_min, x_min, y_max, x_maxdef mkdir(path):	if not os.path.exists(path):		os.mkdir(path)def drawBoundingBoxes(imageData, bbox, classes, color):	"""Draw bounding boxes on an image.	imageData: image data in numpy array format	inferenceResults: inference results array off object (l,t,w,h)	colorMap: Bounding box color candidates, list of RGB tuples.	"""	for res, class_ in zip(bbox, classes):		mask = {0: 'Tank', 1:'Tank Cluster', 2:'Floating Head Tank'}		left = int(res[0])		top = int(res[1])		right = int(res[0]) + int(res[2])		bottom = int(res[1]) + int(res[3])		label = mask[class_]		imgHeight, imgWidth, _ = imageData.shape		#check		thick = max(1, int((res[2] + res[3]) // 200))		cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)		cv2.putText(imageData, label, (left - 12, top + 12), 0, 1e-3 * imgHeight, color, thick//3)	plt.figure(figsize=(12, 12))	plt.imshow(imageData)	plt.grid(False)