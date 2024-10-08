import itertools
from classifier_wrapper import ClassifierWrapper
import pandas as pd
from data_func import DataManager
from name_utils import *

def dict_value_combinations(dicts):
    keys = list(dicts.keys())
    values = list(dicts.values())
    result = []
    for combination in itertools.product(*values):
        result.append(dict(zip(keys, combination)))
    return result


class ClassifierGroup:

    def __init__(self, classifier_classes, param_dicts = None, additional_params=None, dataManager = None):
        if param_dicts is None:
            param_dicts = {}
        if additional_params is None:
            additional_params = {}
        classifiers = []
        self.dataManager = dataManager
        if dataManager is None:
            self.dataManager = DataManager()
        for c in classifier_classes:
            for dict in dict_value_combinations(param_dicts):
                name = name_from_params(c, dict)
                dict.update(additional_params)
                classifiers.append(ClassifierWrapper(c(**dict), name=name, dataManager=self.dataManager))
        self.classifiers = classifiers

    def plot_rocs(self, directory_to_save = None, **kwargs):
        for c in self.classifiers:
            filename = create_file_path(c, "roc", directory_to_save) if directory_to_save else None
            c.plot_roc(file_to_save = filename, **kwargs)

    def plot_confusion_matrices(self, directory_to_save = None, **kwargs):
        for c in self.classifiers:
            filename = create_file_path(c, "confusion_matrix", directory_to_save) if directory_to_save else None
            c.plot_confusion_matrix(file_to_save = filename, **kwargs)

    def plot_learning_curves(self, directory_to_save = None, **kwargs):
        for c in self.classifiers:
            filename = create_file_path(c, "learning_curve", directory_to_save) if directory_to_save else None
            c.plot_learning_curve(file_to_save = filename, **kwargs)

    def calc_params(self, **kwargs):
        return pd.concat([c.calc_params(**kwargs) for c in self.classifiers])

    
        