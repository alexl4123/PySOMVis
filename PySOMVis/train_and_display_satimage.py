import numpy as np
import pandas as pd
import os
from math import sqrt

df = pd.read_csv(os.path.join('datasets', 'satimage', 'satimage.csv'), index_col=False)
arr = df.to_numpy()
arr = np.float32(arr)
slice = int (len(arr)/6)
arr = arr[:slice,:] # TODO: this should be removed?

inputdata = arr[:,:-1]
dimensions = 36
classes = arr[:,-1]
component_names = ['0','1','2','3','4','5','7'] # 6 does not occur

# som_size = int(5*sqrt(len(arr)))
som_size = 60

# Use any library for training SOM map (e.x. MiniSOM, SOMOClu, SOMpy, PopSOM etc.)
from minisom import MiniSom
som = MiniSom(x=som_size,y=som_size,input_len=dimensions)
som.train(inputdata, 100)
weights = som.get_weights()[:,0,:]
for i in range(som_size-1):
    weights = np.concatenate((weights, som.get_weights()[:,i+1,:]), 0) # TODO: this needs to be tested, sorted by features not instacnces, so take first of the 400 then second and so on?

# somuclu doesn't work
# import somoclu
# som = somoclu.Somoclu(som_size, som_size)
# som.train(np.float32(inputdata), epochs=100)



# visualization
from pysomvis import PySOMVis
# pysomviz = PySOMVis(weights=weights, input_data=inputdata)
pysomviz = PySOMVis(weights=weights, m=som_size, n=som_size, dimension=dimensions, input_data=inputdata, classes=classes, component_names=component_names)

pysomviz._onbigscreen()