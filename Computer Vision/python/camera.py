import cv2 as cv
import numpy as np
def image():
    img= cv.imread('./456587258_8502144919817720_7596256156928509379_n.jpg')
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def video():
    vid = cv.VideoCapture('./WhatsApp Video 2024-12-16 at 9.19.03 PM.mp4')
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        cv.imshow('video', frame)
        if cv.waitKey(20) & 0xFF == ord(' '):
            break
    vid.release()
    cv.destroyAllWindows()
def camera():
    cam = cv.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv.imshow('camera', frame)
        if cv.waitKey(20) & 0xFF == ord(' '):
            break
    cam.release()
    cv.destroyAllWindows()
def resizimg():
    img= cv.imread('./456587258_8502144919817720_7596256156928509379_n.jpg')
    resizimg = cv.resize(img, (100, 100), interpolation=cv.INTER_CUBIC) 
    cv.imshow('image', img)
    cv.imshow('resizimg', resizimg)

    cv.waitKey(0)
    cv.destroyAllWindows()
    
def drawing():
    img = np.zeros((600,600,3), dtype='uint8')
    cv.imshow('drawing', img)
    img[400:500, 100:200] = (150, 150, 150)
    cv.imshow('color', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
drawing()    
