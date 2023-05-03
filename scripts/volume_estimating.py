import numpy as np
from skimage import filters
from skimage import measure
from skimage import segmentation
from skimage import morphology
from skimage import color
from skimage import io
from matplotlib import pyplot as plt
import cv2


def conv_bbox(box_xyxy):
	xs = np.array([int(box_xyxy[0]), int(box_xyxy[2])])
	ys = np.array([int(box_xyxy[1]), int(box_xyxy[3])])

	x_min = xs.min()
	x_max = xs.max()
	y_min = ys.min()
	y_max = ys.max()

	return y_min, x_min, y_max, x_max


def check_bb(bbox, shape):
	h, w, _ = shape
	for d in bbox:
		if d <= 2 or d >= w-2:
			return False
	return True


def intersection(bb1, bb2):
	y_min1, x_min1, y_max1, x_max1 = bb1
	y_min2, x_min2, y_max2, x_max2 = bb2

	x_left = max(x_min1, x_min2)
	x_right = min(x_max1, x_max2)
	y_top = max(y_min1, y_min2)
	y_bottom = min(y_max1, y_max2)

	intersection = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
	return intersection


def draw_bbox(imageData, bboxes, classes, color):
	"""Draw bounding boxes on an image.
	imageData: image data in numpy array format
	inferenceResults: inference results array off object (l,t,w,h)
	colorMap: Bounding box color candidates, list of RGB tuples.
	"""
	for res, class_ in zip(bboxes, classes):
		left = int(res[1])
		top = int(res[2])
		right = int(res[3])
		bottom = int(res[0])
		label = class_
		imgHeight, imgWidth, _ = imageData.shape
		thick = 3
		cv2.rectangle(imageData, (left, top), (right, bottom), color, thick)
		cv2.rectangle(imageData, (left, top), (right, bottom), [0, 0, 0], 1)
		txt_size = cv2.getTextSize(class_, 0, 1, 1)
		cv2.rectangle(imageData, (left + 3, top), (int(left + txt_size[0][0]), int(top - txt_size[0][1])),
					  [255, 255, 255], -1)
		cv2.putText(imageData, label, (left + 3, top), 0, 1, [0, 0, 0], 1)


class Tank:
	def __init__(self, box_xyxy, image, log, factor_x=0.5, factor_y=0.6):
		self.image = image
		self.log = log
		self.gt_coords = conv_bbox(box_xyxy)  # bounding box coordinates
		y_min, x_min, y_max, x_max = self.gt_coords
		self.log.info(f"___________Start processing oil tank {[y_min, x_min, y_max, x_max]}_________")
		# scale for tank cropping
		margin_x = int((x_max - x_min) * factor_x)
		margin_y = int((y_max - y_min) * factor_y)

		# y_min, y_max, x_min, x_max values for cropping
		self.y_min = max(y_min - margin_y, 0)
		self.y_max = max(y_max + int(margin_y // 2), 0)
		self.x_min = max(x_min - margin_x, 0)
		self.x_max = max(x_max + margin_x, 0)
		# actual margins, given that the calculated margin might extend beyond the image
		margin_y_true = y_min - self.y_min
		margin_x_true = x_min - self.x_min

		# coordinates of the actual bounding box relative to the crop box
		self.bbox_relative = [margin_y_true, margin_x_true, (y_max - y_min) + margin_y_true,
							  (x_max - x_min) + margin_x_true]
		# crop section of the image
		self.tank_crop = self.image[self.y_min:self.y_max, self.x_min:self.x_max, :]
		self.proc_tank()
		self.get_regions()
		self.log.info(f"___________Successful processing oil tank {[y_min, x_min, y_max, x_max]}_________")

	def proc_tank(self):
		# HSV conversion
		hsv = color.rgb2hsv(self.tank_crop)
		H = hsv[:, :, 0]
		S = hsv[:, :, 1]
		V = hsv[:, :, 2]
		# LAB conversion
		lab = color.rgb2lab(self.tank_crop)
		l1 = lab[:, :, 0]
		l2 = lab[:, :, 1]
		l3 = lab[:, :, 2]
		# Enhanced image
		self.tank_hsv = -(l1 + l3) / (V + 1)
		# Threshold values
		t1 = filters.threshold_minimum(self.tank_hsv)
		t2 = filters.threshold_mean(self.tank_hsv)

		# Thresholding
		self.tank_thresh = self.tank_hsv > (0.6 * t1 + 0.4 * t2)
		# Check for searching the better thresholding
		self.check_threshold()
		# Processed, labeled image
		self.label_image = measure.label(cv2.medianBlur(morphology.area_closing(morphology.closing(
			segmentation.clear_border(filters.hessian(
				self.tank_thresh, mode='constant')))).astype('uint8'), 7))

	def check_threshold(self):
		if self.tank_thresh.sum() > 3000:
			thresh_func = (filters.threshold_isodata,
						   filters.threshold_li,
						   filters.threshold_otsu,
						   filters.threshold_yen)
			score = []
			for i in thresh_func:
				score.append((self.tank_hsv > i(self.tank_hsv)).sum())
			best_thresh = score.index(min(score))
			if score[best_thresh] < 3000:
				self.log.info(f'Default threshold is pass. Using {str(thresh_func[best_thresh]).split(" ")[1]}')
				self.tank_thresh = self.tank_hsv > thresh_func[best_thresh](self.tank_hsv)

	def get_regions(self):
		# Regions within image
		self.regions_all = measure.regionprops(self.label_image)

		self.regions = []

		# Some regions are noise. This ensures that regions have a decent area ( > 25 px),
		# that the region intersects the boudning box around the tank (removes lots of noisy features)
		# and that the processed region is also present in the thresholded image (the hessian filter can sometimes
		# add artifacts that need to be removed this day)
		for region in self.regions_all:
			if intersection(self.bbox_relative, region.bbox) > 300:
				if region.area > 25:
					b = region.bbox
					if abs(self.tank_thresh[b[0]:b[2], b[1]:b[3]].mean() - region.image.mean()) < 0.08:
						self.regions.append(region)

		# areas of all regions
		areas = np.array([i.area for i in self.regions])

		# if there are more than two areas found, take the two largest
		# 1 - ratio of the two largest areas calculates the volume estimation
		if len(areas) > 1:
			idx2, idx1 = areas.argsort()[-2:]
			self.volume = 1 - self.regions[idx2].area / self.regions[idx1].area
			self.log.info(f"Volume = {self.volume}")
		# if only 1 area is found, tank is assumed to be full
		else:
			idx2 = 0
			idx1 = 0
			self.volume = 1
			self.log.info("One area was found. Volume = 1")

		# Blank image onto which to paste only the two shadow regions
		self.blank = np.zeros(self.tank_crop.shape[:2])

		for region in [self.regions[idx1], self.regions[idx2]]:
			y_min, x_min, y_max, x_max = region.bbox
			self.blank[y_min:y_max, x_min:x_max] += region.image.astype('uint8')

		# get contours of shadows
		self.contours = measure.find_contours(self.blank, 0.5)
		if len(self.contours) > 1:
			# If there are multiple contours, take the two longest
			contour_idxs = np.array([len(i) for i in self.contours]).argsort()[-2:]
		else:
			contour_idxs = [0]
		self.contours_select = [self.contours[i] for i in contour_idxs]

	def plot_tank(self):
		fig, axes = plt.subplots(3, 3, figsize=(12, 12))

		fig.suptitle(f'Tank volume: {self.volume:.3f}%')

		axes[0][0].imshow(self.tank_crop)
		axes[0][0].set_title('Tank Crop')

		axes[0][1].imshow(self.tank_crop)
		axes[0][1].imshow(self.blank, alpha=0.5)
		axes[0][1].set_title('Shadow Overlay')

		axes[0][2].imshow(self.tank_crop)
		for cnt in self.contours_select:
			axes[0][2].plot(cnt[:, 1], cnt[:, 0])
		axes[0][2].set_title('Shadow Contour')

		axes[1][0].imshow(self.blank)
		axes[1][0].set_title('Shadow')

		axes[1][1].imshow(np.zeros(self.blank.shape))
		for cnt in self.contours:
			axes[1][1].plot(cnt[:, 1], cnt[:, 0])
		axes[1][1].set_title('All Contours')

		axes[1][2].imshow(np.zeros(self.blank.shape))
		for cnt in self.contours_select:
			axes[1][2].plot(cnt[:, 1], cnt[:, 0])
		axes[1][2].set_title('Major Contours')

		axes[2][0].imshow(self.tank_hsv)
		axes[2][0].set_title('HSV Ratio')

		axes[2][1].imshow(self.tank_thresh)
		axes[2][1].set_title('Tank Thresholding')

		axes[2][2].imshow(self.label_image)
		axes[2][2].set_title('Morphology Labeling')
		for ax in axes.flat:
			ax.axis('off')


class MultiTank:
	def __init__(self, bbs, image, image_path, log):
		self.image = image
		self.image_path = image_path
		self.log = log
		# check bounding boxes aren't at the edge of the image
		self.log.info(f'___________Start processing image: {image_path}_________\n')
		self.bbs = [i for i in bbs if check_bb(i, image.shape)]
		self.tanks = []
		for i in self.bbs:
			try:
				self.tanks.append(Tank(i, image, self.log))
			except IndexError:
				self.log.error("Can't extract shadow")
		self.create_masks()

	def plot_volumes(self, target_path, figsize=(12, 12), ax=None):
		fig, ax = plt.subplots(figsize=figsize)
		coords = [i.gt_coords for i in self.tanks]
		labels = [f'{i.volume:.3f}' for i in self.tanks]
		draw_bbox(self.image, coords, labels, [255, 255, 255])
		io.imsave(f'runs/detect/{target_path}/{self.image_path}', self.image)
		ax.imshow(self.image)
		ax.axis('off')

	def plot_contours(self, figsize=(12, 12)):
		fig, ax = plt.subplots(figsize=figsize)
		ax.imshow(self.image)

		colors = np.linspace(0, 1, len(self.tanks))

		for i, tank in enumerate(self.tanks):
			for contour in tank.contours_select:
				ax.plot(contour[:, 1] + tank.x_min, contour[:, 0] + tank.y_min, color=plt.cm.rainbow(colors[i]))

	def create_masks(self):
		mask = np.zeros(self.image.shape[:2])
		colors = np.linspace(0, 1, len(self.tanks))

		for i, tank in enumerate(self.tanks):
			tank_blank = (tank.blank > 0) * (i + 1)
			mask[tank.y_min:tank.y_max, tank.x_min:tank.x_max] += tank_blank

		self.mask = mask
		self.mask_binary = mask > 0