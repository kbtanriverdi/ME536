import train

import pandas as pd
import numpy as np
import os

files=["classifiervectors.csv","temp_image.csv","temppath.txt","mapping.txt"]
for file in files:
    try:
        os.remove(file)
        print(f"{file} has been deleted.")
    except Exception as e:
        print(f"Exception: {e} while {file}")
        continue
train.train()