# import the necessary packages
import numpy as np
import cv2
from datetime import datetime
 
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)
 
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def saveimage(sufix, img):
    fname = ("%d-%d-%d" % (datetime.now().minute,datetime.now().second,datetime.now().microsecond))
    cv2.imwrite("../gambar/out/"+fname+'-'+sufix+".png",img)

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 15.0
 
# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
KNOWN_WIDTH = 10.0
 
# initialize the list of images that we'll be using
dir = "../gambar/distance"
IMAGE_PATHS = [dir+"/ft2.jpg", dir+"/ft3.jpg", dir+"/ft4.jpg"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (300,400))
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	cm = 30.48 * inches
 
	# draw a bounding box around the image and display it
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fcm" % (cm / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 2)
	cv2.imshow("image", image)
	saveimage("dist",image)
	cv2.waitKey(0)
