"""
PySOMVis is an open-source Python-based GUI implementation for analyzing trained SOMs. It provides a wide range of different visualizations for the SOM, such as Chessboard Visualization, Clustering approach, Component Plane, D-Matrix, Hit Histogram, Metro Map, Neighborhood Graphs, Pie Chart, Smoothed Data Histogram, U-Matrix, U*-Matrix, P-Matrix, Quantization Error, SOMStreamVis
"""

__version__ = '0.0.1'
__author__ = 'Sergei Mnishko'


from .pysomvis import PySOMVis
from .SOMToolBox_Parse import SOMToolBox_Parse

__all__ = [PySOMVis.__name__, SOMToolBox_Parse.__name__]



