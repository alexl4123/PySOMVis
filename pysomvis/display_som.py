import os

from SOMToolBox_Parse import SOMToolBox_Parse
dataset = 'chainlink' #iris, chainlink, 10clusters, BostonHousing
inputdata = SOMToolBox_Parse(os.path.join('datasets', dataset, dataset + '.vec')).read_weight_file()
weights = SOMToolBox_Parse(os.path.join('datasets', dataset, dataset + '.wgt.gz')).read_weight_file()
classinfo = SOMToolBox_Parse(os.path.join('datasets', dataset, dataset+'.cls')).read_weight_file()

# Visualization by PySOMVis
from pysomvis import PySOMVis

vis = PySOMVis(weights=weights['arr'], m=weights['ydim'],n=weights['xdim'],
                dimension=weights['vec_dim'], input_data=inputdata['arr'],
                classes=classinfo['arr'][:,1], component_names=classinfo['classes_names'])
vis._onbigscreen()