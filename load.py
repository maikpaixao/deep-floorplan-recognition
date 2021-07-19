from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class PredictionConfig(Config):
	NAME = "kangaroo_cfg"
	NUM_CLASSES = 1 + 1
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('./doors_model.h5', by_name=True)

image = cv2.imread('image.jpg')
scaled_image = mold_image(image, cfg)
sample = expand_dims(scaled_image, 0)
yhat = model.detect(sample, verbose=0)[0]

pyplot.subplots()
pyplot.imshow(image)
pyplot.title('Predicted')
ax = pyplot.gca()

for box in yhat['rois']:
  print(list(box))
  #y1, x1, y2, x2 = box

#pyplot.show()