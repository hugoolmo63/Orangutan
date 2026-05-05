import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure

def extract_border_features(mask):
    # 1. Basic properties
    # Ensure mask is boolean for accurate counting
    binary_mask = mask > 0
    area = np.count_nonzero(binary_mask)
    
    # 2. Precise Perimeter calculation
    perimeter = measure.perimeter(binary_mask)
    
    # 3. Compactness (Polsby-Popper)
    # Score of 1.0 is a perfect circle. Irregular borders approach 0.0.
    # Formula: (4 * pi * Area) / Perimeter^2
    compactness = (4 * np.pi * area) / (perimeter**2)
    
    # 4. Convexity
    # Measures how 'smooth' or 'bulging' the border is vs 'indented'.
    coords = np.transpose(np.nonzero(binary_mask))
    if len(coords) > 2:
        hull = ConvexHull(coords)
        # In SciPy 2D ConvexHull, .volume is the Area.
        convex_hull_area = hull.volume 
        convexity = area / convex_hull_area
    else:
        convexity = 1.0

    return {
        "compactness": round(compactness, 4),
        "convexity": round(convexity, 4),
    }
    
#Low Compactness: Indicates a highly jagged or "stellate" border
#Low Convexity: Indicates significant "ins and outs" (invaginations) along the border, which is a common red flag for malignancy