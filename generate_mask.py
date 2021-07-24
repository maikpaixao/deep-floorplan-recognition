import os
import argparse
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave, imresize
from matplotlib import pyplot as plt
import cv2

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#parser = argparse.ArgumentParser()
#parser.add_argument('--im_path', type=str, default='./teste/45765448.jpg', help='input image paths.')

floorplan_map = {
	0: [0,0,0], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def saveImage2(image, path):
	plt.gca().set_axis_off()
	plt.figure(figsize=(512/100, 512/100), dpi=100)
	plt.imshow(image/255.)
	plt.savefig(path, bbox_inches='tight',transparent=True, pad_inches=0)

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb
	return rgb_im

def saveImage(image, path, door=False):
	floorplan = image.copy()
	if door==True:
		floorplan[image==1] = 9
		floorplan[image==2] = 10
		image = ind2rgb(floorplan)
	else:
		image = ind2rgb(floorplan)
	imsave(path, image)

def main(args):
	im = imread(args.im_path, mode='RGB')
	im = im.astype(np.float32)
	im = imresize(im, (512,512,3)) / 255.
	#im = im/255

	with tf.Session() as sess:
		sess.run(tf.group(tf.global_variables_initializer(),
					tf.local_variables_initializer()))

		saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan)
		saveImage(room_type, './teste/room_type.png')

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
