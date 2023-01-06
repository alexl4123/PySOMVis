import pandas as pd
import numpy as np
import gzip
import os

from pathlib import Path
from io import StringIO


class SOMToolBox_Parse:
    
    def __init__(self, filename):
        self.filename = filename
    
    def read_weight_file(self,):
        df = {}
        if self.filename[-3:len(self.filename)] == '.gz':
            with gzip.open(self.filename, 'rb') as file:
                df = self._read_vector_file_to_df(df, file)
               #df['arr'] = np.rot90(df['arr'].reshape(df['ydim'],df['xdim'],df['vec_dim'])).reshape(-1,df['vec_dim']) #rotate matrix because of SOMToolbox format

        else:
            with open(self.filename, 'rb') as file:
                df = self._read_vector_file_to_df(df, file)

        file.close()

        return df
    
    def _read_vector_file_to_df(self, df, file):
        for byte in file:
            line = byte.decode('UTF-8')
            if line.startswith('$'):
                df = self._parse_vector_file_metadata(line, df)
            else:
                c = df['vec_dim'] if 'vec_dim' in df else 2 
                if 'arr' not in df: df['arr'] = np.empty((0,c), dtype=float)
                df = self._parse_weight_file_data(line, df)
        return df


    def _parse_weight_file_data(self, line, df):
        splitted=line.split(' ')
        try:
            c = df['vec_dim'] if 'vec_dim' in df else 2
            df['arr'] = np.append(df['arr'], [np.array(splitted[0:c]).astype(float)], axis=0)
        except: raise ValueError('The input-vector file does not match its unit-dimension.') 
        return  df

    def _parse_vector_file_metadata(self, line, df):
        splitted = line.strip().split(' ')
        if splitted[0]   == '$XDIM':        df['xdim']          = int(splitted[1])
        elif splitted[0] == '$YDIM':        df['ydim']          = int(splitted[1])
        elif splitted[0] == '$VEC_DIM':     df['vec_dim']       = int(splitted[1])
        elif splitted[0] == '$CLASS_NAMES': df['classes_names'] = splitted[1:] 
        return df 

    @classmethod
    def write_som_to_files(cls, input_data, class_info, weights, path):
        """
            input_data  - The input data without the class labels -> pandas.DataFrame or numpy.ndarray are supported
            class_info  - The column of the class labels of the input data -> pandas.DataFrame or numpy.ndarray are supported
            weights     - The weights of the trained SOM -> pandas.DataFrame or numpy.ndarray are supported
            path        - A python os path object without the file ending, e.g. os.path.join("datasets","my_data") -> then three files "my_data.vec", "my_data.wgt.gz" and "my_data.cls" are generated
        """

        np_input_data = cls._parse_to_numpy(input_data, "input_data")
        np_class_info = cls._parse_to_numpy(class_info, "class_info")
        np_weights = cls._parse_to_numpy(weights, "weights")

        cls._generate_vector_file(np_input_data, path.with_suffix(".vec"))
        cls._generate_class_file(np_class_info, path.with_suffix(".cls"))
        cls._generate_weights_file(np_weights, path.with_suffix(".wgt.gz"))

    @classmethod
    def _parse_to_numpy(cls, data, name):
        np_data = None

        if data is not None and isinstance(data, pd.DataFrame):
            np_data = data.to_numpy()
        elif data is not None and isinstance(data, np.ndarray):
            np_data = data
        else:
            raise NotImplementedError(name + " may either be a pandas.DataFrame or a numpy.ndarray!")

        return np_data



    @classmethod
    def _generate_vector_file(cls, data, path):

        type_var = "vec"
        x_dim = str(data.shape[0])
        y_dim = str(1)
        vec_dim = str(data.shape[1])

        write_string = ""
        write_string += f"$TYPE {type_var}\n"
        write_string += f"$XDIM {x_dim}\n"
        write_string += f"$YDIM {y_dim}\n"
        write_string += f"$VEC_DIM {vec_dim}\n"

        iteration = 1

        for row in data:
            row_list = list(row)
            
            row_str = ""
            for item in row_list:
                row_str += str(item) + " "
            row_str += str(iteration) + "\n"

            write_string += row_str

            iteration += 1

        with open(path, "w") as output_file:
            output_file.write(write_string)
        output_file.close()   

    @classmethod
    def _generate_class_file(cls, data, path):

        classes = list(np.unique(data))

        type_var = "class_information"
        num_classes = str(len(classes))
        class_names = ""
        for cls in classes:
            class_names += str(cls) + " "
        x_dim = 2
        y_dim = len(list(data))

        write_string = ""
        write_string += f"$TYPE {type_var}\n"
        write_string += f"$NUM_CLASSES {num_classes}\n"
        write_string += f"$CLASS_NAMES {class_names}\n"
        write_string += f"$YDIM {y_dim}\n"

        iteration = 1

        for value in data:
            row_str = str(iteration) + " "
            row_str += str(value) + " "
            row_str += "\n"

            write_string += row_str

            iteration += 1


        with open(path, "w") as output_file:
            output_file.write(write_string)

        output_file.close()   

    @classmethod
    def _generate_weights_file(cls, data, path):

        type_var = "som"
        x_dim = str(data.shape[0])
        y_dim = str(data.shape[1])
        vec_dim = str(data.shape[2])
        z_dim = str(1)

        write_string = ""
        write_string += f"$TYPE {type_var}\n"
        write_string += f"$XDIM {x_dim}\n"
        write_string += f"$YDIM {y_dim}\n"
        write_string += f"$ZDIM {z_dim}\n"
        write_string += f"$VEC_DIM {vec_dim}\n"

        iteration = 1

        for y_index in range(data.shape[1]):

            for x_index in range(data.shape[0]):
                
                row_str = ""

                for z_index in range(data.shape[2]):
        
                    item = data[x_index, y_index, z_index]
                    row_str += str(item) + " "
    
                row_str += "SOM_MAP_" + str(path.name) + "_" + "(" + str(x_index) + "/" + str(y_index) + "/0)\n"
                write_string += row_str

        with open(path, mode="wb") as output_file:
            output_file.write(gzip.compress(bytes(write_string, 'utf-8')))

        output_file.close()

       
