import pdb
import cv2
import os
from collections import OrderedDict
import sys

import numpy as np
# from werkzeug.utils import secure_filename
# from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from PIL import Image as imgpil
import base64
import io
import random

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import std_msgs.msg
from std_msgs.msg import String
import tf
import roslib

from options.test_options import TestOptions
import models
import torch

opt = TestOptions().parse()
model = models.create_model(opt)
model.eval()

# max_size = 256
# max_num_examples = 200

class filler_node:

    def __init__(self):

        self.image = None
        self.mask = None
        self.rate = rospy.Rate(15) 
        self.cvbridge = CvBridge()

        self.msg_header = std_msgs.msg.Header()
        self.depth_msg_header = std_msgs.msg.Header()

        self.image_pub = rospy.Publisher("image_filled", Image, queue_size = 1)

        self.image_sub = rospy.Subscriber("/image_compressed",Image, self.callback)
        print("subscribed to /image_compressed")
        self.depth_sub = rospy.Subscriber("dynamic_mask", Image, self.callback_m)
        print("subscribed to /dynamic_mask")

    def callback(self, frame):
        self.image = self.cvbridge.imgmsg_to_cv2(frame,"bgr8")
            
    def callback_m(self, mask):
        self.mask = self.cvbridge.imgmsg_to_cv2(mask)

    def process_image(self, img, mask):

        # img = imgpil.fromarray(img)
        # img =img.convert("RGB")
        # img_raw = np.array(img)
        # w_raw, h_raw = img.size
        # h_t, w_t = h_raw//8*8, w_raw//8*8

        # img = img.resize((w_t, h_t))
        img = np.array(img).transpose((2,0,1))

        # mask_raw = np.array(mask)[...,None]>0
        # mask = mask.resize((w_t, h_t))

        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = (torch.Tensor(mask)>0).float()
        
        img = (torch.Tensor(img)).float()
        img = (img/255-0.5)/0.5
        img = img[None]
        mask = mask[None,None]

        with torch.no_grad():
            generated,_ = model({'image':img,'mask':mask}, mode='inference')
        generated = torch.clamp(generated, -1, 1)
        generated = (generated+1)/2*255
        generated = generated.cpu().numpy().astype(np.uint8)
        generated = generated[0].transpose((1,2,0))
        # result = np.zeros([generated.shape[0], generated.shape[1], generated.shape[2]])
        # for ch in range(generated.shape[2]):
        #     result[:, :, ch] = np.where(np.asarray(mask) == 1, 0, generated[:, :, ch]) + np.where(np.asarray(mask) == 0, 1, img_raw[:, :, ch])
             
        # result = np.array(result.astype(np.uint8))
        # print('********************************')
        # print(result.shape)

        # result = imgpil.fromarray(result).resize((w_raw, h_raw))
        # result = np.array(result)
        # result = imgpil.fromarray(result.astype(np.uint8))
        cv2.imshow('filled', generated)
        cv2.waitKey(5)
        return generated

    def start(self):
        # rospy.spin()

        while not rospy.is_shutdown():
            if self.image is not None:

                image = Image()
                image = self.process_image(self.image, self.mask)
                image = self.cvbridge.cv2_to_imgmsg(image, "bgr8")
                image.header = self.msg_header
                self.image_pub.publish(image)

            # rospy.loginfo('publishing filled dynamic image')

            self.rate.sleep()


def main(args):
    '''Initializes and cleans up ros node'''
    rospy.init_node('filler_node', anonymous=True)
    node = filler_node()
    node.start()

if __name__ == "__main__":
    main(sys.argv)

