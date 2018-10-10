import sys

# Import PretrainedLoader from the Mask_RCNN directory
# TODO: make this work more generally
MASKRCNN_DIR = '/Users/abba/projects/segmentation/Mask_RCNN'
sys.path.append(MASKRCNN_DIR)
from PretrainedLoader import get_pretrained_model

# instantiate COCO-pretrained model
model, class_names = get_pretrained_model()

# test with provided images!
import cv2
import os
import time
from mrcnn import visualize

# walk directory, time and visualize the model outputs
for root, dirs, files in os.walk("images/", topdown=False):
   for name in files:
       fname = os.path.join("images/", name)
       img = cv2.imread(fname)
       start = time.time()
       results = model.detect([img])
       stop = time.time()
       print('detection time %.5f' % (stop - start))
       r = results[0]
       vis = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
