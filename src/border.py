from skimage import morphology

def border(mask) :
    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(5)
    # Use this "brush" to erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter_im = mask - mask_eroded
    border_ratio = perimeter_im/mask

    return border_ratio