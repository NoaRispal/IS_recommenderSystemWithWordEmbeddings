import numpy as np

def normalize(data: np.array) -> np.array:
    if data.size == 0:
        return data
        
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    
    diff = col_max - col_min
    
    diff[diff == 0] = 1.0
    
    return (data - col_min) / diff
    
    