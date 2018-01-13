import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob


# global params
SIZE = 320      # resize image size
SZ, SV = 25, 0  # FFT mask filter size and value
REQ_RAT = 0.6   # object detection minimum ratio
MSZ = 7        # morfological kernel size

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
        img = cv.resize(img,(640,480))
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


    # fourier transform -- select feature
    f = np.fft.fft2(edges)
    fshift = np.fft.fftshift(f)

    # masking
    rows, cols = edges.shape
    crow,ccol = int(rows/2) , int(cols/2)
    
    # fshift[0:rows-1, 0:cols-1] = SV
    # fshift[crow-SZ:crow+SZ,ccol-SZ:ccol+SZ] = SV
    fshift[0:crow-SZ, 0:ccol-SZ] = SV                 # top-left
    fshift[0:crow-SZ, ccol+SZ:cols-1] = SV            # top-right
    fshift[crow+SZ:rows-1, 0:ccol-SZ] = SV            # bottom-left    
    fshift[crow+SZ:rows-1, ccol+SZ:cols-1] = SV       # bottom-right
    
    fshift[0:crow-SZ, ccol:ccol+SZ] = SV                 # top-center
    fshift[crow+SZ:rows-1, ccol-SZ:ccol] = SV            # bottom-center
    fshift[crow-SZ:crow+SZ, 0:ccol-SZ] = SV            # mid-left    
    fshift[crow-SZ:crow+SZ, ccol+SZ:cols-1] = SV       # mid-right

    magnitude_spectrum = np.log(np.abs(fshift))

    # invert FT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # threshold
    res = np.round(img_back)
    ret,res = cv.threshold(res,127,255,cv.THRESH_BINARY)

    # dilation
    kernel = np.ones((MSZ,MSZ),np.uint8)
    res = cv.dilate(res,kernel,iterations =1)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)

    if debug :
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
        plt.subplot(247),plt.imshow(img_back, cmap = 'gray')
        plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
        plt.subplot(248),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.subplot(339)
        # # plt.hist(fshift.ravel(),256,[0,256])
        # plt.plot(fshift,'r')
        # plt.title('Magnitude Histogram'), plt.xticks([]), plt.yticks([])

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    
        plt.show()          # DISABLE in VIDEO MODE
    return res



### MAIN ####

## ANALYZE SET OF IMAGES
dir = '../gambar/krl/'
r,c = 3,3
for imgs in glob.glob(dir+"/*.png"):
    img0 = scale_down( cv.imread(imgs))
    img = cv.cvtColor(img0,cv.COLOR_RGB2GRAY)
    img = process(img, debug=True)    
    img = devide_into_sec(img,r,c)
    
    # DETECTION
    rows,cols = img.shape
    print(cols, rows)
    fnd = False
    a,b = (0,0),(0,0)
    for y in range(r-1,-1,-1):
        for x in range(c):
            x1,y1,x2,y2 = (int(x*cols/c)+1, int(y*rows/r)+1, int((x+1)*cols/c)-1, int((y+1)*rows/r)-1)
            sec_area = (x2-x1)*(y2-y1)
            tot = np.sum(img[y1:y2,x1:x2])
            rat = tot/sec_area/255
            print(x+1, y+1, (x1,y1,x2,y2), rat, tot, sec_area)
            if(rat>REQ_RAT):
                fnd=True
                if(a==(0,0)): a,b=(x1,y1),(x2,y2)
                if(b<(x2,y2)): b=(x2,y2)

        if fnd: 
            cv.rectangle(img0,a,b,(0,255,0),2) 
            break

    cv.imshow('Processed',img)
    cv.imshow('Original',img0)
    if cv.waitKey() & 0xFF == ord('x'):
        break


# ## ANALYZE VIDEO / CAM
# cap = cv.VideoCapture("../../videos/dramaga.mp4")
# # cap = cv.VideoCapture("http://nasrul_hamid:16017@192.168.100.2:4747/mjpegfeed?320x240")

# r,c = 3,3
# while(True):
#     # Capture frame-by-frame
#     ret, img0 = cap.read()
#     img0 = scale_down(img0)
#     img = cv.cvtColor(img0,cv.COLOR_RGB2GRAY)

#     img = process(img, 1)
#     img = devide_into_sec(img,c,r)

#     # DETECTION
#     rows,cols = img.shape
#     print(rows,cols)
#     for y in range(r):
#         for x in range(c):
#             x1,y1,x2,y2 = (int(x*cols/c)+1, int(y*rows/r)+1, int((x+1)*cols/c)-1, int((y+1)*rows/r)-1)
#             sec_area = (x2-x1)*(y2-y1)
#             tot = np.sum(img[y1:y2,x1:x2])
#             rat = tot/sec_area/255
#             print(y+1, x+1, (x1,y1,x2,y2), rat, tot, sec_area)
#             if(rat> REQ_RAT):
#                 cv.rectangle(img0,(x1,y1),(x2,y2),(0,255,0),2)
    
#     # # Display the resulting frame
#     cv.imshow('processed',img)
#     cv.imshow('original',img0)
    
#     if cv.waitKey(1000) & 0xFF == ord('x'):
#         break

# # When everything done, release the capture
# cap.release()


cv.destroyAllWindows()