from ultralytics import YOLO
import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils
if __name__ == '__main__':
    # Load a model
    model = YOLO("C:/Users/snk20/Downloads/Brain Tumor/runs/classify/train10/weights/best.pt")  # load a custom model
    def crop_img(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = 0
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        
        return new_img
    image = cv2.imread("C:/Users/snk20/Downloads/glinew.jpg")
    new_img = crop_img(image)
    new_img = cv2.resize(new_img,(256,256))
    # Predict with the model
    results = model(new_img)  # predict on an image 