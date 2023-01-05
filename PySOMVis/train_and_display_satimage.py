import numpy as np
import pandas as pd
import os
from math import sqrt

df = pd.read_csv(os.path.join('datasets', 'satimage', 'satimage.csv'), index_col=False)
arr = df.to_numpy()

inputdata = arr[:,:-1]
dimensions = 36
classes = arr[:,-1]
component_names = [0,1,2,3,4,5,7] # 6 does not occur

som_size = int(5*sqrt(len(arr)))

# Use any library for training SOM map (e.x. MiniSOM, SOMOClu, SOMpy, PopSOM etc.)
from minisom import MiniSom
som = MiniSom(x=som_size,y=som_size,input_len=dimensions)
som.train(inputdata, 10)


# visualization
from pysomvis import PySOMVis
pysomviz = PySOMVis(weights=som.get_weights(), input_data=inputdata)
# pysomviz = PySOMVis(weights=som.get_weights(), m=som_size, n=som_size, dimension=dimensions, input_data=inputdata, classes=classes, component_names=component_names)

pysomviz._onbigscreen()