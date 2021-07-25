
import numpy as np
import cv2
import imutils
from process import Process
from PIL import Image
import json
import argparse
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
import generate_mask
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class NumpyEncoder(json.JSONEncoder):
      def default(self, obj):
            if isinstance(obj, np.ndarray):
                  return obj.tolist()
            elif isinstance(obj, np.int32):
                  return obj.tolist()
            return json.JSONEncoder.default(self, obj)

class PredictionConfig(Config):
      NAME = "kangaroo_cfg"
      NUM_CLASSES = 1 + 1
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1

cfg = PredictionConfig()
json_dict = {}

_dmodel = MaskRCNN(mode='inference', model_dir='./', config=cfg)
_dmodel.load_weights('./h5_models/doors_model.h5', by_name=True)

_wmodel = MaskRCNN(mode='inference', model_dir='./', config=cfg)
_wmodel.load_weights('./h5_models/windows_model.h5', by_name=True)

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./teste/095.jpg', help='input image paths.')

def main(args):
      generate_mask.main(args)
      process = Process()

      _original_image = cv2.imread(args.im_path)
      original_image = Image.open(args.im_path).convert("RGBA")
      w,h = original_image.size

      image = cv2.imread('./teste/room_type.png')
      image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
      image = cv2.medianBlur(image, 5)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      kernel = np.ones((5,5))
      erod = cv2.erode(gray, kernel, iterations = 2)
      erod = cv2.dilate(erod, kernel, iterations = 2)

      ret, thresh = cv2.threshold(erod, 127, 255, cv2.THRESH_BINARY_INV)
      cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      cnts = imutils.grab_contours(cnts)
      cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

      image_cpy = image.copy()
      count = 0
      for cnt in cnts[1:]:
            comodo_dict = {}
            comodo = []
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            if(int(cv2.contourArea(cnt)) > 60):
                  comodos_image = Image.open("./teste/room_type.png").convert("RGBA")
                  comodos_image = comodos_image.resize((w,h), Image.ANTIALIAS)

                  color_img = process.crop_box(cnt, np.array(comodos_image), margin=0)
                  text_img = process.crop_box(cnt, np.array(original_image), margin=0)
            
                  room_name = process.find_room_name(color_img)
                  dimensions = process.extract_text(text_img)
                  dimensions = process.format_dimensions(dimensions)

                  doors_bbs = process.detect_doors(_original_image, _dmodel, cfg)
                  doors_list = []
                  for bb in doors_bbs:
                        width, height = bb[3] - bb[1], bb[2] - bb[0]

                        points_doors = [(bb[1], bb[0]), (bb[1]+width, bb[0]), 
                                          (bb[3], bb[2]), (bb[3]-width, bb[2])]

                        doors = process.find_door_within(image_cpy, points_doors, cnt, dimensions)
                        if len(doors)>0:
                              doors_list.append(np.array(doors))

                  windows_bbs = process.detect_windows(_original_image, _wmodel, cfg)
                  windows_list = []
                  for bb in windows_bbs:
                        width, height = bb[3] - bb[1], bb[2] - bb[0]

                        points_windows = [(bb[1], bb[0]), (bb[1]+width, bb[0]), 
                                          (bb[3], bb[2]), (bb[3]-width, bb[2])]

                        windows = process.find_window_within(image_cpy, points_windows, cnt, dimensions)
                        if len(windows)>0:
                              windows_list.append(np.array(windows))

                  comodo_dict['label'] = room_name
                  comodo_dict['rooms_coordinates'] = np.array(process.to_polygon(approx))
                  comodo_dict['doors'] = doors_list
                  comodo_dict['windows'] = windows_list

                  json_dict[count] = comodo_dict
                  count = count + 1
      
      with open('data.json', 'w') as fp:
            json.dump(json_dict, fp, cls=NumpyEncoder)

if __name__ == '__main__':
      FLAGS, unparsed = parser.parse_known_args()
      main(FLAGS)
