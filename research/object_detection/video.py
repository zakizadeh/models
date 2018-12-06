# Fall detector video class
# Source: camera (int) or file (string)
#
#
# Kim Salmi, kim.salmi(at)iki(dot)fi
# http://tunn.us/arduino/falldetector.php
# License: GPLv3

import time
import cv2
import person
import settings
import webservice
import bs
import cv2
import time
import collections
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
import person
import settings
import webservice
import bs

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

cap = cv2.VideoCapture("video.avi")
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


from utils import label_map_util

from utils import visualization_utils as vis_util
# What model to download.
CWD_PATH = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

class Video:
	def __init__(self):
		self.settings = settings.Settings()
		self.camera = cv2.VideoCapture("video.avi")
		self.bs = bs.Bs()
		self.persons = person.Persons(self.settings.movementMaximum, self.settings.movementMinimum, self.settings.movementTime)
		self.start = time.time()
		self.webservice = webservice.Webservice(self.settings.location, self.settings.phone)
		self.errorcount = 0
		self.alertLog = []
		self.frameCount = 1

	def nextFrame(self):
		grabbed, self.frame = self.camera.read()
		if not grabbed: # eof
			self.destroyNow()
		self.convertFrame()

	def showFrame(self):
		if self.settings.debug:
			cv2.imshow("Thresh", self.thresh)
			#if self.settings.bsMethod == 1:
				#cv2.imshow("backgroundFrame", self.backgroundFrame)
				#cv2.imshow("frameDelta", self.frameDelta)
		cv2.imshow("Feed", self.frame)


	def destroyNow(self):
		self.camera.release()
		cv2.destroyAllWindows()

	def testDestroy(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			self.destroyNow()
			return 1
		else:
			return 0

	def resetBackgroundFrame(self):
		grabbed, self.frame = self.camera.read()
		self.convertFrame()
		self.bs.resetBackgroundIfNeeded(self.frame)
		self.persons = person.Persons(self.settings.movementMaximum, self.settings.movementMinimum, self.settings.movementTime)
		#self.frameCount = 1
		#print 'resetbackgroundFrame'

	def testBackgroundFrame(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord("n"):
			self.bs.deleteBackground()
		#self.resetBackgroundFrame()
		
	def updateBackground(self):
		self.bs.updateBackground(self.frame)

	def testSettings(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord("0"):
			self.settings.minArea += 50
			print "minArea: " , self.settings.minArea
		if key == ord("9"):
			self.settings.minArea -= 50
			print "minArea: " , self.settings.minArea
		if key == ord("8"):
			self.settings.dilationPixels += 1
			print "dilationPixels: " , self.settings.dilationPixels
		if key == ord("7"):
			self.settings.dilationPixels -= 1
			print "dilationPixels: " , self.settings.dilationPixels
		if key == ord("6"):
			self.settings.thresholdLimit += 1
			print "thresholdLimit: " , self.settings.thresholdLimit
		if key == ord("5"):
			self.settings.thresholdLimit -= 1
			print "thresholdLimit: " , self.settings.thresholdLimit
		if key == ord("4"):
			self.settings.movementMaximum += 1
			print "movementMaximum: " , self.settings.movementMaximum
		if key == ord("3"):
			self.settings.movementMaximum -= 1
			print "movementMaximum: " , self.settings.movementMaximum
		if key == ord("2"):
			self.settings.movementMinimum += 1
			print "movementMinimum: " , self.settings.movementMinimum 
		if key == ord("1"):
			self.settings.movementMinimum  -= 1
			print "movementMinimum: " , self.settings.movementMinimum 
		if key == ord("o"):
			if self.settings.useGaussian:
				self.settings.useGaussian = 0
				print "useGaussian: off"
				self.resetbackgroundFrame()
			else:
				self.settings.useGaussian = 1
				print "useGaussian: on"
				self.resetbackgroundFrame()
		if key == ord("+"):
			self.settings.movementTime += 1
			print "movementTime: " , self.settings.movementTime
		if key == ord("p"):
			self.settings.movementTime -= 1
			print "movementTime : " , self.settings.movementTime



	def convertFrame (self):
		# resize current frame, make it gray scale and blur it
		if self.settings.useResize:
			r = 750.0 / self.frame.shape[1]
			dim = (750, int(self.frame.shape[0] * r))
			self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA)
		if self.settings.useBw:
			self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		if self.settings.useGaussian:
			self.frame = cv2.GaussianBlur(self.frame, (self.settings.gaussianPixels, self.settings.gaussianPixels), 0)

	def compare(self):
		# difference between the current frame and backgroundFrame
		self.thresh = self.bs.compareBackground(self.frame)
		self.thresh = cv2.dilate(self.thresh, None, iterations=self.settings.dilationPixels) # dilate thresh
		_, contours, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours
		category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
		self.persons.tick()

		detectStatus = "idle"

                with detection_graph.as_default():
		  with tf.Session(graph=detection_graph) as sess:
		    # Definite input and output Tensors for detection_graph
		    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		    # Each box represents a part of the image where a particular object was detected.
		    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		    # Each score represent how level of confidence for each of the objects.
		    # Score is shown on the result image, together with the class label.
		    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		    ret,image_np=cap.read()
		    # the array based representation of the image will be used later in order to prepare the
		    # result image with boxes and labels on it.
		    #image_np = load_image_into_numpy_array(image)
		    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		    image_np_expanded = np.expand_dims(image_np, axis=0)
		    # Actual detection.
		    (boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})

		    # Create a display string (and color) for every box location, group any boxes
		    # that correspond to the same location.
		    image=image_np
		    im_width, im_height, channels = image.shape
		    boxes=np.squeeze(boxes)
		    classes=np.squeeze(classes).astype(np.int32)
		    scores=np.squeeze(scores)
		    category_index=category_index
		    instance_masks=None
		    keypoints=None
		    use_normalized_coordinates=False
		    max_boxes_to_draw=20
		    min_score_thresh=.5
		    agnostic_mode=False
		    line_thickness=4
		    box_to_display_str_map = collections.defaultdict(list)
		    box_to_color_map = collections.defaultdict(str)
		    box_to_instance_masks_map = {}
		    box_to_keypoints_map = collections.defaultdict(list)
		    if not max_boxes_to_draw:
			max_boxes_to_draw = boxes.shape[0]
		    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > min_score_thresh:
			  box = tuple(boxes[i].tolist())
			  if instance_masks is not None:
			    box_to_instance_masks_map[box] = instance_masks[i]
			  if keypoints is not None:
			    box_to_keypoints_map[box].extend(keypoints[i])
			  if scores is None:
			    box_to_color_map[box] = 'black'
			  else:
			    if not agnostic_mode:
			      if classes[i] in category_index.keys():
				class_name = category_index[classes[i]]['name']
			      else:
				class_name = 'N/A'
			      display_str = '{}'.format(class_name)
			    else:
				 display_str = 'score: {}%'.format(int(100 * scores[i]))
			      #display_str = 'score: {}%'.format(int(100 * scores[i]))


			    if display_str =='person':
				y, x, h, w = box
				person = self.persons.addPerson(x, y, w, h)
				color = (0, 255, 0)
				if person.alert:
				   detectStatus = "Alarm, not moving"
				   cv2.putText(self.frame, "Alarm, not moving:", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 1)
					
				print(box)
				box_to_display_str_map[box].append(display_str)
				if agnostic_mode:
				  box_to_color_map[box] = 'DarkOrange'
				else:
				  box_to_color_map[box] = STANDARD_COLORS[
				      classes[i] % len(STANDARD_COLORS)]
		      # Draw all boxes onto image.
		    for box, color in box_to_color_map.items():
			 ymin, xmin, ymax, xmax = box
			 #print(box)
			 (xminn, xmaxx, yminn, ymaxx) =(xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
			 print(xminn,xmaxx,yminn,ymaxx)

		    vis_util.visualize_boxes_and_labels_on_image_array(
			  image_np,
			  np.squeeze(boxes),
			  np.squeeze(classes).astype(np.int32),
			  np.squeeze(scores),
			  category_index,
			  use_normalized_coordinates=True,
			  max_boxes_to_draw=20,
			  min_score_thresh=.5,
			  line_thickness=8)
		    cv2.imshow('object detection',cv2.resize(image_np,(800,600)))
		
		# Hud + fps
		if self.settings.debug:
			self.end = time.time()
			seconds = self.end - self.start
			fps  = round((1 / seconds), 1)
			self.start = time.time()

			cv2.putText(self.frame, "Status: {}".format(detectStatus), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 1)
	 		cv2.putText(self.frame, "FPS: {}".format(fps), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 1)

	def newLightconditions(self):
		self.errorcount += 1
		if self.errorcount > 10:
	 		time.sleep(1.0)
	 		self.resetBackgroundFrame()
	 		self.errorcount = 0
