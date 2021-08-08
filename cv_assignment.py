import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np

# Variables
# for defining name of images easily
filename1 = "japan.jpeg"                                       # Must have to write names correctly or else it will throw error!
filename1_eq = "japan_eq.jpeg"
filename2 = "japan_new.jpeg"

# Adaptive histogram used to improve the quality
# of image segmentation
img1 = cv2.resize(cv2.imread(filename1, 1), [800,600])          # You can change name of file you want to work
ycrcb_img = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
eq_imgx = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(filename1_eq,eq_imgx)                               # To save histogram equalized image. save with another name

# with pre trained model we can easily classify any given image or video
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
segment_image.segmentImage(filename1_eq, show_bboxes = True, output_image_name = filename2)
img2 = cv2.resize(cv2.imread(filename2, 1), [800,600])      


cv2.imshow('Img1 & img2', np.hstack([eq_imgx,img2]))               # display both images in a single horizontal window
cv2.waitKey(0)
cv2.destroyAllWindows()
