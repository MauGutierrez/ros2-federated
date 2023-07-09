import numpy as np

def serialize_array(arr: list) -> list:
    new_arr = [vector.tolist() for vector in arr]
    
    return new_arr

def deserialize_array(arr: list) -> object:
    new_arr = [np.array(vector, dtype=np.float32) for vector in arr]
    
    return new_arr
