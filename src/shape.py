from skimage import measure
import numpy as np
from scipy.spatial import ConvexHull
import cv2
from math import sqrt, floor, ceil, nan, pi

def border(mask) :
    ''' 
    Computes a border score that evalutes the irregularity level of the contour of the lesion.
    Scale frome 0 to 1 where 1 indicates a smooth border (with a homogenus distance frome the center).

    Args: image (numpy.ndarray), mask of the lession
    Return: border score (float) : Float between 0 and 1 indicating border regularity
    '''
    # Find the contour :
    contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0: # Checking if the contour exist
        return 0.0

    contour = max(contours, key=cv2.contourArea) #keep the biggest contour

    # Chek if the contour is null, to prevent errors :
    if len(contours) == 0:
        return 0.0

    # Compute the center and its coordonates :
    M = cv2.moments(contour) # Compute geometric moments of the contour.

    if M["m00"] == 0: # Security to prevent divsion by 0
        return 0.0
    
    cx = M["m10"] / M["m00"] # Compute x-coordinate of the centroid (center of mass).
    cy = M["m01"] / M["m00"] # Compute y-coordinate of the centroid (center of mass).

    # Comput the distance from center to contour for each point :
    distances = [] # List to store center->perimenter distance
    for point in contour:
        x, y = point[0] #get x,y coordonates for the selected contour pixel
        d = np.sqrt((x - cx)**2 + (y - cy)**2) #compute distance frome center to contour pixel
        distances.append(d)

    distances = np.array(distances) # convert distances to np.array to make computations easyer

    score = 1-(np.std(distances) / np.mean(distances)) # Normalize by mean distance so size does not dominate and make that 1 means regular border

    return float(round(score,3)) 

def compactness(mask):
    '''Computes a compactness score for the given mask.
    The score is based of the Polsby-Popper measure.
    The score falls between the value 0 and 1. Scores closer to 1 indicates a more compact mask.

    Args: mask (numpy.ndarray): input masked image
    Returns: compactness_score (float): Float between 0 and 1 indicating compactness.
    Source: 05_Feature_Extraction
    '''

    mask = mask > 0
     
    area = np.sum(mask) #Area of ground truth

    perimeter = measure.perimeter(mask) #Compute the perimeter of the lesion

    if area == 0 or perimeter == 0 :
        return 0.0

    compactness = (4 * np.pi * area) / (perimeter ** 2)

    score = round(compactness, 3)

    return score

def convexity(mask):
    '''Calculate convexity score between 0 and 1,
    with 1 indicating a smoother border and 0 a more crooked border.

    Args: image (numpy.ndarray): input masked image
    Returns: convexity_score (float): Float between 0 and 1 indicating convexity.
    Source: 05_Feature_Extraction
    '''

    mask = mask > 0

    coords = np.transpose(np.nonzero(mask)) # Get coordinates of all pixels in the lesion mask

    if len(coords) < 3 : # Check if the mask is not to small
        return 0.0
    
    hull = ConvexHull(coords) # Compute convex hull of lesion pixels

    if hull.volume == 0 : # Security to prevent divsion by 0
        return 0.0
    
    lesion_area = np.count_nonzero(mask) # Compute area of lesion mask

    convexity = lesion_area / hull.volume # Compute convexity as ratio of lesion area to convex hull

    return round(convexity, 3)