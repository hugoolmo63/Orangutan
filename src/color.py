#at first we did a function without a saturation feature, but then we realize that is very
#important to medical image analysis (useful for detecting 'blueness' in melanoma)

import numpy as np #working with pixels
import cv2 #loading images
from sklearn.cluster import KMeans #for reduce pixel's weight by identifying dominant colors
from scipy.spatial.distance import pdist #used to instantly calculate the pairwise distance
from skimage.color import rgb2hsv #it converts physical light data (RGB) into perceptual data (HSV)

def analyze_color_profile(image_rgb, mask, n_colors=5):
    if len(mask.shape) == 3: #checking the dimensions of the array
        #standard color image: (1080, 1920, 3) and greyscale: (1080, 1920)
        mask = mask[:, :, 0] #turning to a 2D image.
    
    lesion_pixels_rgb = image_rgb[mask > 0] #selecting pixels that are not black, or in other words, the lesion (white area)
    skin_pixels_rgb = image_rgb[mask == 0] #selecting black pixels, or skin area
    
    if lesion_pixels_rgb.size == 0: #in case we upload a corrupted file
        return None

    lesion_pixels_hsv = rgb2hsv(lesion_pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    #we only covert rgb to hsv the lesion area, saving a lot of memory
    #rgb2hsv needs a shape of 1 pixel high, N pixels wide, and has 3 color channels; then flatten it back into a simple list of px

    rgb_variances = np.var(lesion_pixels_rgb, axis=0) #measuring the variance of colors (the more colors, a higher variance)
    #melanomas are often "variegated" (multicolored), so high variance is a key indicator
    
    #hue in hsv is not a line but a circle (0ºto360º)
    h_sin = np.sin(2 * np.pi * lesion_pixels_hsv[:, 0]) #converting this linear scale into coordenates using cos and sen
    h_cos = np.cos(2 * np.pi * lesion_pixels_hsv[:, 0]) #we multiply by 2pi, converting this value (0 to 1) into radians
    hue_variance = 1 - np.sqrt(np.mean(h_sin)**2 + np.mean(h_cos)**2) #if the length_avg = 1, var = 0 (all colors are identical)
    #otherwise (length = 0, var = 1) the colors are completely dispersed
    
    sat_variance = np.var(lesion_pixels_hsv[:, 1]) #saturation in hsv
    val_variance = np.var(lesion_pixels_hsv[:, 2]) #color value in hsv

    step = max(1, lesion_pixels_rgb.shape[0] // 1000) #selecting n/1000 pixels (usually images contain over 1m)
    sample_pixels = lesion_pixels_rgb[::step] #from start to end stepping 1000 pixels each time
    
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(sample_pixels) #finding the more representative n_colors 
    centers = kmeans.cluster_centers_ #coordenates of that colors (by using kmeans' algorithm)
    max_dist = np.max(pdist(centers)) if len(centers) > 1 else 0 #finding maximum pairwise distance between dominant colors
    #if max_dist is low = homogenious lesion (almost one color); but if = high, lesion is polychromatic (multicolored)

    lesion_mean = np.mean(lesion_pixels_rgb, axis=0) #general tone of the lesion, or avg color
    skin_mean = np.mean(skin_pixels_rgb, axis=0) if skin_pixels_rgb.size > 0 else lesion_mean #avg color of the outside area
    
    return {
        "rgb_variances": rgb_variances,      # [R, G, B]
        "hsv_variances": [hue_variance, sat_variance, val_variance],
        "max_dist": max_dist,                # Multicolor rate
        "dominant_colors": centers,          # RGB Palette
        "contrast_delta": lesion_mean - skin_mean
    }
    
my_image = cv2.imread(input('insert image'))
my_image_rgb = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
my_mask = cv2.imread(input('insert mask'))