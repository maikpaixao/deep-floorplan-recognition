
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
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class PredictionConfig(Config):
      NAME = "kangaroo_cfg"
      NUM_CLASSES = 1 + 1
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1

def detect_doors(images, model):
      points = []
      image = cv2.imread('image.jpg')
      scaled_image = mold_image(image, cfg)
      sample = expand_dims(scaled_image, 0)
      yhat = model.detect(sample, verbose=0)[0]

      for box in yhat['rois']:
            points.append(list(box))
      return points

def detect_windows(images, model):
      points = []
      image = cv2.imread('image.jpg')
      scaled_image = mold_image(image, cfg)
      sample = expand_dims(scaled_image, 0)
      yhat = model.detect(sample, verbose=0)[0]
      
      for box in yhat['rois']:
            points.append(list(box))
      return points

cfg = PredictionConfig()
json_dict = {}

_dmodel = MaskRCNN(mode='inference', model_dir='./', config=cfg)
_dmodel.load_weights('./doors_model.h5', by_name=True)

_wmodel = MaskRCNN(mode='inference', model_dir='./', config=cfg)
_wmodel.load_weights('./windows_model.h5', by_name=True)

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

def main(args):
      process = Process()
      original_image = Image.open("teste/original_5.jpg").convert("RGBA")
      #original_image = Image.open(args).convert("RGBA")
      w,h = original_image.size

      image_entry_path = 'teste/fronteiras_5.png'
      #image_entry_path = args[:-4]+'_room_boundary.png'
      #image_rooms_path = args[:-4]+'_room_type.png'
      image = cv2.imread('teste/comodos_5.png')
      #image = cv2.imread(image_rooms_path)

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

      #process.generate_entries(image_entry_path, w, h)

      #image_doors = cv2.imread(image_entry_path[:-4] + 'entries' + image_entry_path[-4:])
      #points_in_door, points_out_door = process.detect_floor(image_doors) ## door coordinates
      #points_doors = points_in_door + points_out_door

      image_cpy = image.copy()
      count = 0
      for cnt in cnts[1:]:
            comodo_dict = {}
            comodo = []
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            if(int(cv2.contourArea(cnt)) > 60):
                  comodos_image = Image.open("teste/comodos_5.png").convert("RGBA")
                  comodos_image = comodos_image.resize((w,h), Image.ANTIALIAS)

                  color_img = process.crop_box(cnt, np.array(comodos_image), margin=0)
                  text_img = process.crop_box(cnt, np.array(original_image), margin=0)
            
                  room_name = process.find_room_name(color_img)
                  dimensions = process.extract_text(text_img)
                  dimensions = process.format_dimensions(dimensions)

                  doors_bbs = detect_doors(original_image, _dmodel)
                  for bb in doors_bbs:
                        width, height = bb[3] - bb[1], bb[0] - bb[2]

                        points_doors = [[(bb[1], bb[0]), (bb[1]+width, bb[2])], 
                                          [(bb[3], bb[2]), (bb[3]-width, bb[2])]]

                        portas = process.find_door_within(image_cpy, points_doors, cnt, dimensions)
                        comodo.append(portas)

                  wimdows_bbs = detect_doors(original_image, _wmodel)
                  
                  comodo.append(room_name)
                  comodo.append(process.to_polygon(approx))

                  #portas = process.find_door_within(image_cpy, points_doors, cnt, dimensions)
                  #comodo.append(portas)

                  comodo_dict['name'] = str(comodo[0])
                  comodo_dict['rooms'] = str(comodo[1])
                  comodo_dict['doors'] = str(portas)
                  #comodo_dict['windows'] = wimdows_bbs

                  json_dict[count] = comodo_dict
                  count = count + 1
      
      with open('data.json', 'w') as fp:
            json.dump(json_dict, fp)

if __name__ == '__main__':
      FLAGS, unparsed = parser.parse_known_args()
      main(FLAGS.im_path)
