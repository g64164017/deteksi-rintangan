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
cv.createTrackbar("th", "hough", 30, 500, nothing)
cv.createTrackbar("min", "hough", 10, 500, nothing)
cv.createTrackbar("gap", "hough", 10, 500, nothing)
# cv.namedWindow("threshold")
# cv.createTrackbar("thresh", "threshold",80,500, nothing)
# cv.createTrackbar("maxval", "threshold",255,500, nothing)

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

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
 
	# return the edged image
	return edged

# def clustering_lines(lines):
#     for l in lines:

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process_filters(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum
        

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

    # # Gabor Filters
    # filters = build_filters()
    # gray = process_filters(gray, filters)
    # cv.imshow('result', gray)

    # hsv = cv.cvtColor(cimg, cv.COLOR_RGB2HSV );
    # channels = cv.split(hsv);
    # gray = channels[0];
    # cv.imshow("gray",gray)
    gray = cv.equalizeHist(gray)
    # cv.imshow("gray eq",gray)

    # vector<Vec4i> lines;
    # std::vector<int> labels;
    # int numberOfLines = cv::partition(lines, labels, isEqual);

    while(True):
        pimg = img.copy()

        blur_kernel,blur_sigmax=cv.getTrackbarPos("kernel", "blur"),cv.getTrackbarPos("sigmaX", "blur")
        blur = cv.GaussianBlur(gray, (3+2*blur_kernel,3+2*blur_kernel), blur_sigmax)
        cv.imshow("blur",blur)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))
        # mask_closed = cv.morphologyEx(m)
        
        # ## THRESHOLD
        # con_th,con_max=cv.getTrackbarPos("thresh", "threshold"),cv.getTrackbarPos("maxval", "threshold")
        # ret, thresh = cv.threshold(blur, con_th, con_max, 0)
        # cv.imshow("threshold",thresh)

        # ## EDGE: CONTOURS
        # # edges = np.zeros_like(img)
        # im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(pimg, contours, -1, (0,0,255), 2)
        # print("contours = ", len(contours))

        ## LAPLACIAN
        # laplacian = cv.Laplacian(blur, cv.CV_64F)
        # sobelx = cv.Sobel(blur,cv.CV_64F,1,0,ksize=5)
        # sobely = cv.Sobel(blur,cv.CV_64F,0,1,ksize=5)
        # sobelx = np.uint8(sobelx)
        # cv.imshow('laplacian',laplacian)
        # cv.imshow('sobelx',sobelx)
        # cv.imshow('sobely',sobely)
        
        th1,th2=cv.getTrackbarPos("th1", "edges"),cv.getTrackbarPos("th2", "edges")
        # edges = cv.Canny(blur,th1,th2)
        edges = auto_canny(blur)
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
        himg = np.zeros_like(pimg)

        
        try:
            i=0
            m_line = {}
            for line in lines2:
                i+=1
                T=str(i)
                x1,y1,x2,y2 = line[0]
                m_line[i]=round((1.0*(y1-y2)/(x1-x2)),2)
                print(T,line[0],m_line[i])
                if(abs(m_line[i])<=1.0):
                    cv.line(himg,(x1,y1),(x2,y2),(0,255,0),1)
                else:
                    cv.line(himg,(x1,y1),(x2,y2),(255,0,0),1)
                # cv.putText(pimg, T ,(x1,y1), cv.FONT_HERSHEY_SIMPLEX, 1,(256,0,0),2,cv.LINE_AA)
            
            m_line = sorted(m_line.items(), key=lambda x:x[1])
            # print('m_line = ',m_line)
            
        except TypeError:
            print("TypeError: 'NoneType' object is not iterable")

        cv.imshow('edges',edges)
        cv.imshow('hough',himg)
        
        if (cv.waitKey(1) & 0xFF == ord('q') and frameType==0) or frameType == 1:
            break
    
    # saveimage("blur",blur)
    # saveimage("threshold",thresh)
    # saveimage("canny",edges)
    # saveimage("hough",pimg)

    return(pimg)


## ANALYZE SET OF IMAGES
dir = '../gambar/stairs/'
# i,a=0,0
for imgs in glob.glob(dir+"/*.jpg"):

    img = process(cv.imread(imgs))


# # ANALYZE VIDEO / CAM
# cap = cv.VideoCapture("../../videos/krl1.mp4")

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