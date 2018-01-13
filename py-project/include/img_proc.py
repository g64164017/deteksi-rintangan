#!/usr/bin/env python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

# global params
SIZE = 320      # resize image size
SZ, SV = 25, 0  # FFT mask filter size and value
MSZ = 7        # morfological kernel size
HT_ANGLE = 30   # max angle allowed on Hough transform
DEBUG = True   # debug mode

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
    # Fill inside the polygon
    cv.fillPoly(mask, vertices, match_mask_color)
    # Returning the image only where mask pixels match
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def scale_down(img):
    ## DOWN SCALE
    try:
        scale = 1.0*SIZE / max(img.shape)
        # h,w = scale*img.shape[0], scale*img.shape[1]
        img = cv.resize(img,(0,0), fx=scale, fy=scale)
    except AttributeError:
        img = cv.resize(img,(320,240))
    return img

def devide_into_sec (img, r, c):
    # draw line
    rows, cols = img.shape
    for x in range(1,c):
        cv.line(img,(int(cols*x/c),0),(int(cols*x/c),rows-1),(255,255,255),1)
    for x in range(1,r):
        cv.line(img,(0,int(rows*x/r)),(cols-1,int(rows*x/r)),(255,255,255),1)
    return img

def process (img, frameType=0, debug=False):

    # equalized image    
    eq = cv.equalizeHist(img)
    

    # remove noise
    blur = cv.GaussianBlur(eq, (7,7), 0)
    

    # edge detection
    edges = auto_canny(blur)


    ## EDGE: HOUGH TRANSFORM
    lines2 = cv.HoughLinesP(edges, 1, np.pi / 180, 52,
        minLineLength=42,maxLineGap=10)
    candidate = np.zeros_like(img)

    try:
        i=0
        m_line = {}
        for line in lines2:
            i+=1
            T=str(i)
            x1,y1,x2,y2 = line[0]
            m_line[i]=round((1.0*(y1-y2)/(x1-x2)),2)
            # print(T,line[0],m_line[i])
            if(abs(m_line[i])<=math.tan(HT_ANGLE*math.pi/180)):
                cv.line(candidate,(x1,y1),(x2,y2),(255,255,255),1)
            # else:
            #     cv.line(res,(x1,y1),(x2,y2),(255,0,0),1)
            # cv.line(res,(x1,y1),(x2,y2),255,1)
        
    except TypeError:
        print("TypeError: 'NoneType' object is not iterable")

    # dilation
    kernel = np.ones((MSZ,MSZ),np.uint8)
    res = cv.dilate(candidate,kernel,iterations =1)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)

    if DEBUG:
        plt.subplot(241),plt.imshow(img, cmap = 'gray')
        plt.title('Gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(242),plt.hist(img.ravel(),256,[0,256])
        plt.title('Gray Hist'), plt.xticks([]), plt.yticks([])
        plt.subplot(243),plt.imshow(eq, cmap = 'gray')
        plt.title('Equalized'), plt.xticks([]), plt.yticks([])
        plt.subplot(244),plt.hist(eq.ravel(),256,[0,256])
        plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])
        plt.subplot(245),plt.imshow(blur, cmap = 'gray')
        plt.title('Blurred'), plt.xticks([]), plt.yticks([])
        plt.subplot(246),plt.imshow(edges, cmap = 'gray')
        plt.title('Canny'), plt.xticks([]), plt.yticks([])
        plt.subplot(247),plt.imshow(candidate, cmap = 'gray')
        plt.title('Candidates'), plt.xticks([]), plt.yticks([])
        plt.subplot(248),plt.imshow(res, cmap = 'gray')
        plt.title('Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(339)
        # # plt.hist(fshift.ravel(),256,[0,256])
        # plt.plot(fshift,'r')
        # plt.title('Magnitude Histogram'), plt.xticks([]), plt.yticks([])

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    
        plt.show()          # DISABLE in VIDEO MODE

    return res