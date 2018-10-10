#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from mrcnn import PretrainedLoader

global rgb_counter
rgb_counter = -1
depth_counter = 0

global model
global class_name
model, class_name = PretrainedLoader.get_pretrained_model()

img = cv2.imread('rgb/rgb000.png')
results = model.detect([img])
r = results[0]
print(r['masks'])

def callback(data):
    global rgb_counter
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    rgb_counter += 1
    if rgb_counter % 1000 == 0:
        results = model.detect([cv_image])
        r = results[0]
        mask = r['masks']
        print(mask)
    else:
        print(rgb_counter)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)
    # print("color image received")
    # cv2.imwrite('./rgb/rgb'+str(rgb_counter).zfill(3)+'.png',cv_image)
    # cv2.imwrite('./rgb/rgb.png',cv_image)
    
    # rgb_counter += 1
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


def depth_callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
    except CvBridgeError as e:
        print(e)
    # print("depth image received")
    #cv2.imshow("Image window", cv_image)
    cv2.imwrite('./depth/depth'+str(depth_counter).zfill(3)+'.png',cv_image)
    # cv2.imwrite('./depth/depth.png',cv_image)
    #cv2.waitKey(3)
    depth_counter += 1
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/kinect2/hd/image_color", Image, callback)
    # rospy.Subscriber("/kinect2/hd/image_depth_rect", Image, depth_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    global bridge
    bridge = CvBridge()
    listener()
