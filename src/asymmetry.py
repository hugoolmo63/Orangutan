import numpy as np
from skimage.transform import rotate

def get_asymmetry(mask):
    scores = []
    '''
    Measures the level of inequality of both halfs of the mask (as if we fold a paper).
    The more the left side differs from the right side, the greater the resulting
    value will be.
    '''
    
    # Run the analysis at different angles to ensure rotation invariance.
    # (The mask could be vertically symmetric but not horizontally, that's why)
    for _ in range(6):
        summed = np.sum(mask, axis=0) # vertical sum of pixels
        half_sum = np.sum(summed) / 2
        mid = 0
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                mid = i
                break # gravity point: where half of white pixels are in the left
            # and the other half in the right
        
        y_nonzero, x_nonzero = np.nonzero(mask)
        if len(y_nonzero) == 0: # Safety check for empty masks
            return 0
            
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)] # Making a list with the 
        # min/max values of the ROWS where there are pixels
        x_lims_raw = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        # the same with COLUMNS
        
        # Calculate distance to ensure a symmetric horizontal crop around the midpoint
        x_dist = max(np.abs(x_lims_raw - mid))
        x_left = int(mid - x_dist)
        x_right = int(mid + x_dist)
        
        # Extract the symmetric segment
        segment = mask[y_lims[0]:y_lims[1], x_left:x_right]
        
        if np.sum(segment) > 0:
            # XOR compares original pixels with mirrored pixels (np.flip)
            # sumXOR is the 'number of errors' where the mirrored pixels dont match (0-1)
            mismatch = np.sum(np.logical_xor(segment, np.flip(segment, axis=1)))
            scores.append(mismatch / np.sum(segment)) #Equalization / Normalization
        
        # Rotate for the next iteration (30 degrees)
        mask = rotate(mask, 30)
        
    return sum(scores) / len(scores)