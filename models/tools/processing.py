import pdb

import cv2
from PIL import Image, ImageFilter
import numpy as np



def post_processing(image, methods=['median']):
    for method in methods:
        if method.split('_')[0] == 'erode':
            kernel_size = int(method.split('_')[1])
            iteration = int(method.split('_')[2])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.erode(image, kernel, iterations=iteration) 
        if method.split('_')[0] == 'dilate':
            kernel_size = int(method.split('_')[1])
            iteration = int(method.split('_')[2])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.dilate(image, kernel, iterations=iteration) 
        elif method == 'mode':
            f = ImageFilter.ModeFilter(size=7)

        elif method == 'opening':
            kernel = np.ones((3, 3), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif method == 'gaussian':
            f = ImageFilter.GaussianBlur(radius=5)

        elif method == 'sharpening':
            f = ImageFilter.EDGE_ENHANCE_MORE

        elif method == 'median':
            f = ImageFilter.MedianFilter(size=5)
        
        if method in ['mode', 'gaussian', 'sharpening', 'median'] :
            image = Image.fromarray(image).filter(f)


    return image

def rgba2rgb(rgba, background=(0,0,0) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )