import os
import argparse
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave, imresize
from matplotlib import pyplot as plt
import cv2

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

# color map
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
	'''
	fig = plt.figure(frameon=False)
	fig.set_size_inches(512,512)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(image, aspect='auto')
	fig.savefig(path)
	'''
	plt.gca().set_axis_off()
	plt.figure(figsize=(512/100, 512/100), dpi=100)
	#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	#plt.margins(0,0)
	plt.imshow(image/255.)
	plt.savefig(path, bbox_inches='tight',transparent=True, pad_inches=0)

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.iteritems():
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
	
	#image = np.uint8(image)
	#image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	#cv2.imwrite(path, floorplan/255.)
	#cv2.imwrite(path, floorplan)
	#imsave(path, floorplan/255.)
	imsave(path, image)

def main(args):
	# load input
	im = imread(args.im_path, mode='RGB')
	#w,h,_ = im.shape
	im = im.astype(np.float32)
	im = imresize(im, (512,512,3)) / 255.
	#im = im/255

	# create tensorflow session
	with tf.Session() as sess:
		
		# initialize
		sess.run(tf.group(tf.global_variables_initializer(),
					tf.local_variables_initializer()))

		# restore pretrained model
		saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# get default graph
		graph = tf.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan)

		# plot results
		#plt.subplot(121)
		#plt.imshow(im)
		#plt.subplot(122)
		#plt.imshow(floorplan_rgb/255.)
		#plt.imshow(room_type/255.)
		#plt.show()
		
		#plt.savefig('result.jpg')
		saveImage(room_type, 'room_type.png')
		#saveImage2(room_boundary, 'room_boundary.png')
		saveImage(room_boundary, 'room_boundary.png', door=True)

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
