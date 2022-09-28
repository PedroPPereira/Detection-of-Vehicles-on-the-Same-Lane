# -*- coding: utf-8 -*-
#!/usr/bin/python
import cv2 as cv
import time
#-----------------------
import modules.imageForms as iF
import modules.inicialization as init
import modules.lane_enchancement as lane_ench
import modules.hough_transform as hough
import modules.vehicle_detector as veh_det
#-----------------------
bool_stream = False #True #False





### ----------------------------------------------------------------------------------------------------------------------------------------------
def vehicleDetection(gpu, in1, imgRAW):
    imgCopy = imgRAW.copy()
    imgGRAY = cv.cvtColor(imgRAW, cv.COLOR_BGR2GRAY)
    imgBGRA = cv.cvtColor(imgGRAY, cv.COLOR_GRAY2BGRA)
    
    
    # 1 - LANE ENCHANCEMENT (CANNY)
    img_gau = lane_ench.gaussian_blur(gpu, in1, imgBGRA)
    img_sobel, theta = lane_ench.sobel_transform(gpu, in1, img_gau)
    img_canny = lane_ench.non_max_suppression(gpu, in1, img_sobel, theta)
    

    # 2 - HOUGH TRANSFORM
    lin_left, lin_right = hough.hough_transform(gpu, in1, img_canny)
    img_hough, array_line = hough.hough_vote(gpu, lin_left, lin_right, in1, imgRAW)


    # 4 - VEHICLE DETECTOR
    img_viola = veh_det.viola_jones(in1, imgCopy, img_hough, array_line)
    
    #cv.imwrite('savedImage.jpg', img_viola)
    return img_viola
### ----------------------------------------------------------------------------------------------------------------------------------------------




    

### ----------------------------------------------------------------------------------------------------------------------------------------------
#                                   IMAGE 
if (bool_stream):
    #get image
    imgRAW = cv.imread("images/f3.png")
    # 0 - INICIALIZATION 
    in1 = init.Init(imgRAW, bool_stream)
    gpu = iF.OpenCL_init(in1)
    # PERFORM ALGORITHM
    start_time_IMG = time.time()
    imgOut = vehicleDetection(gpu, in1, imgRAW)
    print("- RUN IMAGE --- %s seconds ---" % (time.time() - start_time_IMG))
    iF.showSideBySideImages(imgRAW, imgOut, "Original", "Altered", "VEHICLE DETECTION", False, False)



### ----------------------------------------------------------------------------------------------------------------------------------------------    
#                            VIDEO SEQUENCE    
else:
    #get image
    vidCap = cv.VideoCapture("images/video1.MTS")
    # 0 - INICIALIZATION 
    in1 = init.Init("", bool_stream)
    gpu = iF.OpenCL_init(in1)
    # PERFORM ALGORITHM
    while(True):
        ret, vidFrame = vidCap.read()
        if (not ret):
            break
     
        imgOut = vehicleDetection(gpu, in1, vidFrame)
            
        cv.imshow("Video", imgOut)
        if (cv.waitKey(1) >= 0):
            break
        
        
        
        
        
        
        
        
        
        