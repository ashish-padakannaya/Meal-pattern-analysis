import pandas as pd
from helper import get_meal_vectors
import os
import numpy as np
from plots import makePlots

if __name__ == "__main__":
    
    #get patient dataframe from CSV
    data, labels = get_meal_vectors()
    print("data shape is ", data.shape)



