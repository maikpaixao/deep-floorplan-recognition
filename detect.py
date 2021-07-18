
import numpy as np
import cv2
import imutils
from process import Process
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

json_dict = {}

def main(args):
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

      process = Process()
      process.generate_entries(image_entry_path, w, h)

      image_doors = cv2.imread(image_entry_path[:-4] + 'entries' + image_entry_path[-4:])
      points_in_door, points_out_door = process.detect_floor(image_doors) ## door coordinates
      points_doors = points_in_door + points_out_door

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
                  
                  comodo.append(room_name)
                  comodo.append(process.to_polygon(approx))

                  portas = process.find_door_within(image_cpy, points_doors, cnt, dimensions)
                  comodo.append(portas)

                  comodo_dict['nome'] = str(comodo[0])
                  comodo_dict['paredes'] = str(comodo[1])
                  comodo_dict['portas'] = str(portas)

                  json_dict[count] = comodo_dict
                  count = count + 1
      
                  #print(comodo_dict)
      with open('data.json', 'w') as fp:
            json.dump(json_dict, fp)

if __name__ == '__main__':
      FLAGS, unparsed = parser.parse_known_args()
      main(FLAGS.im_path)
