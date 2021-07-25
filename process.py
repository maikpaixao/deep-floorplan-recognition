import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from scipy.spatial import distance as dist
import re
from PIL import Image, ImageDraw
import imutils
import pytesseract
from pytesseract import Output
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
import json

class Process:
  def __init__(self):
    self.data = []
  
  def is_horizontal(self, p1, p2):
      if abs(p2[0] - p1[0]) <= abs(p2[1] - p1[1]):
            return False
      else:
            return True

  def mean_point(self, p1, p2):
      if abs(p2[0] - p1[0]) <= abs(p2[1] - p1[1]):
            return (p2[0], int((p2[1] + p1[1])/2))
      else:
            return (p2[1], int((p2[0] + p1[0])/2))

  def max_point(self, door_points):
      maximum_x = 0
      maximum_y = 0
      for points in door_points:
            if points[0] >= points[1]:
                  if points[0] > maximum_x:
                        maximum_x = points[0]
            else:
                  if points[1] > maximum_y:
                        maximum_y = points[1]
      return (maximum_x, 0), (maximum_y, 1)
  
  def to_polygon(self, approx):
      polygon = []
      for element in approx:
            point_squeezed = np.squeeze(np.array(element))
            point = (point_squeezed[0], point_squeezed[1])
            polygon.append(point)
      return polygon
  
  def crop_image(self, im, approx):
      imArray = np.asarray(im)

      polygon  = self.to_polygon(approx)
      maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
      ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
      mask = np.array(maskIm)

      newImArray = np.empty(imArray.shape,dtype='uint8')
      newImArray[:,:,:3] = imArray[:,:,:3]
      newImArray[:,:,3] = mask*255

      newIm = Image.fromarray(newImArray, "RGBA")
      return newIm

  def extract_text(self, newIm):
      config = '--psm 1 --oem 3'
      data = pytesseract.image_to_data(newIm, config=config, lang='eng', output_type=Output.DICT)
      conf = np.asarray(data['conf']).astype('int8')

      mean_conf = 0
      if len(conf[conf > 0]) > 1:
            mean_conf = np.mean(conf[conf > 0])
      elif len(conf[conf > 0]) == 1:
            mean_conf = conf[conf > 0][0]

      text_ = np.asarray(data['text'])[conf > 0]
      
      dimensions = ''
      for t in text_:
            if 'm' in list(t) or 'cm' in list(t):
                  if len(t) > 1:
                        dimensions += t + ' '
      return dimensions

  def top_left(self, box):
        least_x = 1000
        top_y = 1000
        for point in box:
              if point[0] <= least_x:
                  least_x = point[0]

              if point[1] <= top_y:
                    top_y = point[1]
        return (least_x, top_y)
  
  def top_right(self, box):
        least_x = 0
        top_y = 1000
        for point in box:
              if point[0] >= least_x:
                  least_x = point[0]

              if point[1] <= top_y:
                    top_y = point[1]
        return (least_x, top_y)

  def bottom_left(self, box):
        least_x = 1000
        top_y = 0
        for point in box:
              if point[0] <= least_x:
                  least_x = point[0]

              if point[1] >= top_y:
                    top_y = point[1]
        return (least_x, top_y)

  def bottom_right(self, box):
        least_x = 0
        top_y = 0
        for point in box:
              if point[0] >= least_x:
                  least_x = point[0]

              if point[1] >= top_y:
                    top_y = point[1]
        return (least_x, top_y)

  def crop_box(self, cnt, img, margin=5):
    x,y,w,h = cv2.boundingRect(cnt)
    y1 = y - margin
    x1 = x - margin
    x2 = x + w + margin
    y2 = y + h + margin

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        y2 = img.shape[0]

    cropped = img[y1 : y2, x1 : x2]
    return cropped
      
  def find_room_name(self, image):
      rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

      kitchen_color = np.array([224,255,192])
      closet_color = np.array([192,192,224])
      bathroom_color = np.array([192,255,255])
      bedroom_color = np.array([255,224,128])
      hall_color = np.array([255,160,96])
      balcony_color = np.array([255,224,224])

      bathroom_mask = cv2.inRange(rgb, bathroom_color, bathroom_color)
      closet_mask = cv2.inRange(rgb, closet_color, closet_color)
      kitchen_mask = cv2.inRange(rgb, kitchen_color, kitchen_color)
      bedroom_mask = cv2.inRange(rgb, bedroom_color, bedroom_color)
      balcony_mask = cv2.inRange(rgb, balcony_color, balcony_color)
      hall_mask = cv2.inRange(rgb, hall_color, hall_color)

      if np.sum(bathroom_mask) > 0:
            return 'bathroom'
      elif np.sum(closet_mask) > 0:
            return 'closet'
      elif np.sum(kitchen_mask) > 0:
            return 'livingroom/kitchen/dining_room'
      elif np.sum(bedroom_mask) > 0:
            return 'bedroom'
      elif np.sum(balcony_mask) > 0:
            return 'balcony'
      elif np.sum(hall_mask) > 0:
            return 'hall'
      else:
            return 'not identified'

  def find_doors_contours(self, cnts, image_cpy):
    points_in_door = []
    points_out_door = []

    for cnt in cnts: #doors
      peri = cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

      box = cv2.minAreaRect(approx)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")

      tl = self.top_left(box)
      tr = self.top_right(box)
      bl = self.bottom_left(box)
      br = self.bottom_right(box)
      
      if(int(cv2.contourArea(cnt)) > 5):
            if self.is_horizontal(tl, br):
                  points_in_door.append([tl,tr])
                  points_out_door.append([bl,br])
            else:
                  points_in_door.append([tl,bl])
                  points_out_door.append([tr,br])

    return points_in_door, points_out_door
  
  def detect_floor(self, image):
    image = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3))
    erod = cv2.dilate(gray, kernel, iterations = 1)

    cnts = cv2.findContours(erod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #doors
    
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=False)
    image_cpy = image.copy()
    return self.find_doors_contours(cnts, image_cpy)

  def format_dimensions(self, dimensions):
      dims = []
      dimensions = dimensions.replace("\n", "")
      dimensions = dimensions.split(" ")
      for dimension in dimensions:
            if len(dimension) > 1:
                  dimension_aux = re.sub("[()/!.,]", "", dimension)
                  dims.append(dimension_aux)
      return dims

  def format_roomname(self, room_name):
      room = []
      if len(room_name) > 1:
            room_name_aux = re.sub("[()/!., \n]", "", room_name)
            room.append(room_name_aux)
      return room

  def visualize(self, image_cpy):
      plt.imshow(image_cpy)
      plt.show()

  def relative_distance(self, contour, point):
      M = cv2.moments(contour)
      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])
      center = (cx, cy)
      mean = ((int(point[0][0]) + int(point[1][0]))/2, (int(point[1][0]) + int(point[1][1]))/2)
      return center, mean

  def find_window_within(self, image_cpy, windows_doors, contour, dimensions):
      windows = []
      dist_1 = cv2.pointPolygonTest(contour, windows_doors[0], False)
      dist_2 = cv2.pointPolygonTest(contour, windows_doors[1], False)
      dist_3 = cv2.pointPolygonTest(contour, windows_doors[2], False)
      dist_4 = cv2.pointPolygonTest(contour, windows_doors[3], False)
            
      windows_dict = {}

      if dist_1 > 0 or dist_2>0 or dist_3>0 or dist_4>0:
            if len(dimensions)>1:
                  if self.is_horizontal(windows_doors[0], windows_doors[1]):
                        windows_dict['coordinates'] = windows_doors
                        windows_dict['reference'] = dimensions[1]
                  else:
                        windows_dict['coordinates'] = windows_doors
                        windows_dict['reference'] = dimensions[0]
            
            else:
                  center, mean = self.relative_distance(contour, [windows_doors[0], windows_doors[3]])
                  _ref = None
                  if mean[0] > center[0]:
                        _ref = 'rooms_left'
                  elif mean[0] < center[0]:
                        _ref = 'rooms_right'
                  else:
                        _ref = 'rooms_center'

                  windows_dict['coordinates'] = windows_doors
                  windows_dict['reference'] = _ref

      #if len(windows_dict)>0:
            #windows.append(windows_dict)
      return windows_dict

  def find_door_within(self, image_cpy, points_doors, contour, dimensions):
      doors = []
      dist_1 = cv2.pointPolygonTest(contour, points_doors[0], False)
      dist_2 = cv2.pointPolygonTest(contour, points_doors[1], False)
      dist_3 = cv2.pointPolygonTest(contour, points_doors[2], False)
      dist_4 = cv2.pointPolygonTest(contour, points_doors[3], False)
            
      doors_dict = {}

      if dist_1 > 0 or dist_2>0 or dist_3>0 or dist_4>0:
            if len(dimensions)>1:
                  if self.is_horizontal(points_doors[0], points_doors[1]):
                        doors_dict['coordinates'] = points_doors
                        doors_dict['reference'] = dimensions[1]
                  else:
                        doors_dict['coordinates'] = points_doors
                        doors_dict['reference'] = dimensions[0]
            
            else:
                  center, mean = self.relative_distance(contour, [points_doors[0], points_doors[3]])
                  _ref = None

                  if mean[0] > center[0]:
                        _ref = 'rooms_left'
                  elif mean[0] < center[0]:
                        _ref = 'rooms_right'
                  else:
                        _ref = 'rooms_center'

                  doors_dict['coordinates'] = points_doors
                  doors_dict['reference'] = _ref

      #if len(portas_dict)>0:
            #portas.append(portas_dict)
      return doors_dict

  def order_points_old(self, origin, point):
    refvec = [0, 1]
    vector = [point[0]-origin[0], point[1]-origin[1]]
    lenvector = math.hypot(vector[0], vector[1])
    
    if lenvector == 0:
        return -math.pi, 0
        
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0] * refvec[0] + normalized[1]*refvec[1]
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
    angle = math.atan2(diffprod, dotprod)
    
    if angle < 0:
        return 2*math.pi+angle, lenvector
    
    return angle, lenvector

  def detect_doors(self, image, model, cfg):
      points = []
      #image = cv2.imread('image.jpg')
      scaled_image = mold_image(image, cfg)
      sample = expand_dims(scaled_image, 0)
      yhat = model.detect(sample, verbose=0)[0]

      for box in yhat['rois']:
            points.append(list(box))
      return points

  def detect_windows(self, image, model, cfg):
      points = []
      #image = cv2.imread('image.jpg')
      scaled_image = mold_image(image, cfg)
      sample = expand_dims(scaled_image, 0)
      yhat = model.detect(sample, verbose=0)[0]
      
      for box in yhat['rois']:
            points.append(list(box))
      return points

  def generate_entries(self, path,w,h):
      image = cv2.imread(path)
      image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
      image = cv2.medianBlur(image, 5)
      
      grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      kernel = np.ones((4,4),np.uint8)
      grayImage = cv2.dilate(grayImage, kernel, iterations = 2)

      ret, thresh = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
      walls = cv2.bitwise_not(thresh)

      entries = cv2.bitwise_and(grayImage, grayImage, mask = walls)
      entries = cv2.medianBlur(entries, 5)

      cv2.imwrite(path[:-4] + 'entries' + path[-4:], entries)

