#!/usr/bin/env python
import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
import glob
import math
import sys
sys.path.append('include')
import pytone   as tn
import img_proc as imp

REQ_RAT = 0.2   # object detection minimum ratio
PLAY_MODE = 0   # 0 for image; 1 for video
cap = cv.VideoCapture("../../videos/dramaga.mp4")
# cap = cv.VideoCapture("http://nasrul_hamid:16017@192.168.100.2:4747/mjpegfeed?320x240")
img_files = '../gambar/stairs/*.jpg'
r,c = 3,3


### MAIN ####


if PLAY_MODE == 0:
    ## ANALYZE SET OF IMAGES
    for imgs in glob.glob(img_files):
        img0 = imp.scale_down( cv.imread(imgs))
        img = cv.cvtColor(img0,cv.COLOR_RGB2GRAY)
        img = imp.process(img)    
        img = imp.devide_into_sec(img,r,c)
        
        # DETECTION
        rows,cols = img.shape
        print(cols, rows)
        fnd = False
        a,b = (0,0),(0,0)
        for y in range(r-1,-1,-1):
            for x in range(c):
                x1,y1,x2,y2 = (
                    int(x*cols/c)+1, int(y*rows/r)+1, 
                    int((x+1)*cols/c)-1, int((y+1)*rows/r)-1
                    )
                sec_area = (x2-x1)*(y2-y1)
                tot = np.sum(img[y1:y2,x1:x2])
                rat = tot/sec_area/255
                rat1 = REQ_RAT*math.log(y+2)
                print(x+1, y+1, (x1,y1,x2,y2), rat, tot, sec_area,rat1)
                if(rat>rat1):
                    fnd=True
                    if(a==(0,0)): a,b=(x1,y1),(x2,y2)
                    if(b<(x2,y2)): b=(x2,y2)

            if fnd: 
                cv.rectangle(img0,a,b,(0,255,0),2) 
                tn.play_tone(y)
                break

        cv.imshow('Processed',img)
        cv.imshow('original',img0)
        # thread1.join()
        if cv.waitKey() & 0xFF == ord('x'):
            break

elif PLAY_MODE == 1:
    ## ANALYZE VIDEO / CAM

    while(True):
        # Capture frame-by-frame
        ret, img0 = cap.read()
        img0 = imp.scale_down(img0)
        img = cv.cvtColor(img0,cv.COLOR_RGB2GRAY)

        img = imp.process(img, 1)
        img = imp.devide_into_sec(img,c,r)

        # DETECTION
        rows,cols = img.shape
        print(cols, rows)
        fnd = False
        a,b = (0,0),(0,0)
        for y in range(r-1,-1,-1):
            for x in range(c):
                x1,y1,x2,y2 = (
                    int(x*cols/c)+1, int(y*rows/r)+1, 
                    int((x+1)*cols/c)-1, int((y+1)*rows/r)-1
                    )
                sec_area = (x2-x1)*(y2-y1)
                tot = np.sum(img[y1:y2,x1:x2])
                rat = tot/sec_area/255
                rat1 = math.log(y+3)
                print(x+1, y+1, (x1,y1,x2,y2), rat, tot, sec_area,REQ_RAT/rat1)
                if(rat>REQ_RAT/rat1):
                    fnd=True
                    if(a==(0,0)): a,b=(x1,y1),(x2,y2)
                    if(b<(x2,y2)): b=(x2,y2)

            if fnd: 
                cv.rectangle(img0,a,b,(0,255,0),2) 
                tn.play_tone(y)
                break
        
        # # Display the resulting frame
        cv.imshow('processed',img)
        cv.imshow('original',img0)
        
        if cv.waitKey(1000) & 0xFF == ord('x'):
            break

    # When everything done, release the capture
    cap.release()


cv.destroyAllWindows()