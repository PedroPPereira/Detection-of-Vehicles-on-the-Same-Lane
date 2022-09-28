# -*- coding: utf-8 -*-

import cv2 as cv



def line_equation(array_line, xx):
    x1, y1, x2, y2 = array_line
    try:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1 
    except ZeroDivisionError:
        m = 0
        b = 0
    return m*xx + b
    


def viola_jones(in1, img_cars, img_sobrepose, array_line):
    
    haar = cv.CascadeClassifier(in1.classifier)
    #returns a list of rectangles that are listed as car. 
    cars = haar.detectMultiScale(                              
                img_cars,
                scaleFactor = in1.scale_factor,
                minSize = in1.min_size,
                minNeighbors= in1.minNeighbors,
                maxSize = (img_cars.shape[0] // 2, 
                           img_cars.shape[1] // 2) 
                )
    
    for (x,y,w,h) in cars:
        #ignore the inferior part of the image
        if (y+h/2 > in1.highWindow):
            continue
        #see if car centroid is below both lines
        if (line_equation(array_line[:4], x+w/2) < y+h/2 and line_equation(array_line[4:8], x+w/2) < y+h/2):
            valueRGB = (0, 0, 255)
        else:
            valueRGB = (0, 255, 0)
        img_sobrepose = cv.rectangle(img_sobrepose, (x,y), (x+w, y+h), valueRGB, 4)    

    return img_sobrepose



