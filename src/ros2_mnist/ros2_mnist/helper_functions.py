import numpy as np
import json

def serialize_array(arr):
    new_arr = [vector.tolist() for vector in arr]
    
    return new_arr

def deserialize_array(arr):
    new_arr = [np.array(vector, dtype=np.float32) for vector in arr]
    
    return new_arr
