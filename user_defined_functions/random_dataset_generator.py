# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:22:00 2026

@author: 14062
"""

import numpy as np
import pandas as pd


def dataset_generator():
    print('##### DATASET GENERATOR #####')
    
    ### user input
    rg_seed = int(input('Enter random seed: '))
    lower_bound = float(input('Enter lower bound: '))
    upper_bound = float(input('Enter upper bound: '))
    dim_input = input('Enter dimensions of the data array (comma-separated): ')
    dimensions = tuple(int(dim.strip()) for dim in dim_input.split(','))
    dist_type = input("Choose distribution (type 'uniform' or 'normal'): ").strip().lower()
    
    ### random number generator
    rg = np.random.default_rng(rg_seed)
    
    ### if logic
    if dist_type == 'uniform':
        data = rg.uniform(lower_bound, upper_bound, size=dimensions)
        
    elif dist_type == 'normal':
        mean = float(input("Enter mean for normal distribution: "))
        std_dev = float(input("Enter standard deviation: "))
        data = rg.normal(mean, std_dev, size=dimensions)
        
    else:
        raise ValueError("Invalid distribution choice. Please type 'uniform' or 'normal'.")
        
    ### adding column names
    num_cols = dimensions[1] if len(dimensions) > 1 else 1
    col_input = input(f"Enter variable names for {num_cols} column(s) (comma-separated): ")
    col_names = [col.strip() for col in col_input.split(',')]
    
    ### check for correct number of columns
    if len(col_names) != num_cols:
        print(f"\n[Warning]: Expected {num_cols} names but got {len(col_names)}. Using default numbers.")
        df = pd.DataFrame(data)
        
        
    df = pd.DataFrame(data, columns=col_names)

    return df


df = dataset_generator()
df