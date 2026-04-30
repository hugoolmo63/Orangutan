import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_img() :
    img_name = input('pleas enter the img name with no formating information')
    img_path = '../data/imgs/'+img_name+'.png'
    img_org = cv2.imread(img_path) # loads the image in BGR format
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY) # convert to grayscale for processing
    return img_org, img_gray

def hair_coverage(img_gray):
    '''
    Estimate the proportion of hair in a grayscale image.(from exercice FYP2026_06_shortcuts.ipynb)

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Ratio of pixels detected as hair (between 0 and 1).
    '''

    # generate hair mask using BlackHat filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = cv2.countNonZero(hair_mask)
    coverage = hair_area/total_area
    
    return round(coverage, 4)

def removeHair_auto(img_org, img_gray):
    '''
    Automatically remove hair from an image by adjusting parameters
    based on estimated hair coverage. (from exercice FYP2026_06_shortcuts.ipynb)

    Parameters
    ----------
    img_org : np.ndarray
        Original color image.
    img_gray : np.ndarray
        Grayscale version of the image.

    Returns
    -------
    blackhat : np.ndarray
        Image highlighting detected hair.
    mask : np.ndarray
        Binary mask of hair regions.
    img_out : np.ndarray
        Image with hair removed.
    '''
    # Calculate hair coverage
    coverage = hair_coverage(img_gray)

    # Set parameters based on coverage
    if coverage < 0.05 :
        kernel_size = 1
        threshold = 20
        radius = 1 
    elif 0.05 < coverage < 0.1 :
        kernel_size = 3
        threshold = 17
        radius = 2
    elif 0.1 < coverage < 0.15 :
        kernel_size = 6
        threshold = 11
        radius = 3
    elif 0.15 < coverage < 0.2 :
        kernel_size = 9
        threshold = 8
        radius = 4

    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair contours in preparation for the inpainting algorithm
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, mask, radius, cv2.INPAINT_TELEA)

    return blackhat, mask, img_out

if __name__ == "__main__":
    img_org, img_gray = load_img()
    print("Image loaded!")
    print("Shape:", img_org.shape)