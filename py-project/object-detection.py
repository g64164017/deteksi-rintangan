import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

SIZE=600

def nothing(x): pass
cv.namedWindow("blur")
cv.createTrackbar("kernel", "blur", 2, 4, nothing)
cv.createTrackbar("sigmaX", "blur", 0, 100, nothing)
cv.namedWindow("edges")
cv.createTrackbar("th1", "edges", 100, 500, nothing)
cv.createTrackbar("th2", "edges", 300, 500, nothing)
cv.namedWindow("hough")
cv.createTrackbar("teta", "hough", 10, 20, nothing)
cv.createTrackbar("th", "hough", 52, 500, nothing)
cv.createTrackbar("min", "hough", 42, 500, nothing)
cv.createTrackbar("gap", "hough", 10, 500, nothing)
cv.namedWindow("threshold")
cv.createTrackbar("thresh", "threshold",80,500, nothing)
cv.createTrackbar("maxval", "threshold",255,500, nothing)

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

def saveimage(sufix, img):
    fname = ("%d-%d-%d" % (datetime.now().minute,datetime.now().second,datetime.now().microsecond))
    cv.imwrite("../gambar/out/"+fname+'-'+sufix+".png",img)

def process (img, frameType=0):
    ## PREPARATION: RESIZE IMAGE
    try:
        scale = 1.0*SIZE / max(img.shape)
        h,w = scale*img.shape[0], scale*img.shape[1]
        img = cv.resize(img,(0,0), fx=scale, fy=scale)
    except AttributeError:
        img = cv.resize(img,(640,480))
    cimg=img.copy()
    
    # ## PREPARATION: MASKING INTEREST REGION
    # # shape mask
    # region_of_interest_vertices = [
    #     (0  , h / 3),
    #     (w, h / 3),
    #     (w, h),
    #     (0  , h),
    # ]
    # cimg = region_of_interest(
    #     img,
    #     np.array([region_of_interest_vertices], np.int32),
    # )
    # # plt.figure()
    # # plt.imshow(cropped_image)

    ## PREPARATION: ENHANCEMENT
    gray = cv.cvtColor(cimg,cv.COLOR_BGR2GRAY)
    # cv.imshow("gray",gray)
    gray = cv.equalizeHist(gray)
    # cv.imshow("gray eq",gray)

    while(True):
        pimg = img.copy()

        blur_kernel,blur_sigmax=cv.getTrackbarPos("kernel", "blur"),cv.getTrackbarPos("sigmaX", "blur")
        blur = cv.GaussianBlur(gray, (3+2*blur_kernel,3+2*blur_kernel), blur_sigmax)
        cv.imshow("blur",blur)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))
        # mask_closed = cv.morphologyEx(m)
        
        ## THRESHOLD
        con_th,con_max=cv.getTrackbarPos("thresh", "threshold"),cv.getTrackbarPos("maxval", "threshold")
        ret, thresh = cv.threshold(blur, con_th, con_max, 0)
        cv.imshow("threshold",thresh)

        ## EDGE: CONTOURS
        # edges = np.zeros_like(img)
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(pimg, contours, -1, (0,0,255), 2)
        print("contours = ", len(contours))

        th1,th2=cv.getTrackbarPos("th1", "edges"),cv.getTrackbarPos("th1", "edges")
        edges = cv.Canny(blur,th1,th2)
        # print("canny edges = ", edges)

        # ## CONTOUR RECTANGLE
        # for cnt in contours:
        #     rect = cv.minAreaRect(cnt)
        #     box = cv.boxPoints(rect)
        #     box = np.int0(box)
        #     cv.drawContours(pimg,[box],0,(255,255,0),2)

        ## EDGE: HOUGH TRANSFORM
        hough_teta,hough_th,hough_min,hough_gap = cv.getTrackbarPos("teta", "hough"),cv.getTrackbarPos("th", "hough"),cv.getTrackbarPos("min", "hough"),cv.getTrackbarPos("gap", "hough")
        lines2 = cv.HoughLinesP(edges, 1, 0.1 * hough_teta * np.pi / 180, hough_th,
            minLineLength=hough_min,maxLineGap=hough_gap)
        try:
            i=0
            for line in lines2:
                i+=1
                T=str(i)
                x1,y1,x2,y2 = line[0]
                print(T,line[0],(0.1*(y1-y2)/(x1-x2)))
                cv.line(pimg,(x1,y1),(x2,y2),(0,255,0),2)
                # cv.putText(pimg, T ,(x1,y1), cv.FONT_HERSHEY_SIMPLEX, 1,(256,0,0),2,cv.LINE_AA)
        except TypeError:
            print("TypeError: 'NoneType' object is not iterable")

        cv.imshow('edges',edges)
        cv.imshow('hough',pimg)
        
        if (cv.waitKey(1) & 0xFF == ord('q') and frameType==0) or frameType == 1:
            break
    
    saveimage("blur",blur)
    saveimage("threshold",thresh)
    saveimage("canny",edges)
    saveimage("hough",pimg)

    return(pimg)


## ANALYZE SET OF IMAGES
dir = '../gambar/krl/'
# i,a=0,0
for imgs in glob.glob(dir+"/*.png"):

    img = process(cv.imread(imgs))


# # ANALYZE VIDEO / CAM
# cap = cv.VideoCapture("../videos/bs-in1.mp4")

# while(True):
#     # Capture frame-by-frame
#     ret, img = cap.read()

#     img = process(img, 1)

#     # # Display the resulting frame
#     # cv.imshow('frame',img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()


cv.destroyAllWindows()