import numpy as np
import pandas as pd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

class DTW_wrapper():
    
    def get_selected_class_set(self, _class_to_check: int, how_many_N: int):
        data = pd.read_csv(f"mitbih_test_33_class_{_class_to_check}.csv")
        data = data.drop(data.columns[0], axis=1).drop(data.columns[-1], axis=1).to_numpy()
        return data[:how_many_N]


    def calculate_distance_vector_between_sets(self,_class_to_check: int, new_vector: np.ndarray[float], N: int):
        result = np.array([])
        print(f"selected class to check with DTW: {_class_to_check}")
        data = self.get_selected_class_set(_class_to_check, N)
        print(data)
        for class_vector in data:
            result = np.append(result, np.array([self.calculate_distance_between_vectors(class_vector,new_vector)]))
        return result 

    def Knn(self, N, result_class0 = [], result_class1 = [], result_class2 = [], result_class3 = [], result_class4 = []):  
        for result0 in result_class0:
            


    def calculate_distance_between_vectors(self,vector_1,vector_2):
        return dtw.distance(vector_1, vector_2)

