import pandas as pd
from os.path import abspath, exists
from os import remove
from imblearn.over_sampling import RandomOverSampler, SMOTE

from consts import CLASS_COL, VALUE_COLS

DATA_PATH = abspath("./mitbih_train_33.csv")
TEST_PATH = abspath("./DTW_test_data.csv")
def balance_classes(data, sampling):
    if sampling == 'RandomOverSampler':
        sampler = RandomOverSampler(k_neighbors=3)
    elif sampling == 'SMOTE':
        sampler = SMOTE(k_neighbors=3)
    X_resampled, y_resampled = sampler.fit_resample(data[VALUE_COLS], data[CLASS_COL])
    data[VALUE_COLS] = X_resampled
    data[CLASS_COL] = y_resampled
    return data




def load_data(path, imputation, sampling):
    data = pd.read_csv(path)
    # skip unamed colunamedumn form csv
    data = data.drop(data.columns[0], axis=1)
    data = imputate_data(data, imputation)
    if sampling is not None:
        data = balance_classes(data, sampling)
    print(data.columns.tolist())
    return data


def imputate_data(data, imputation):
    if imputation == None:
        return data
    data[VALUE_COLS] = pd.DataFrame(imputation.fit_transform(data[VALUE_COLS]),columns = data[VALUE_COLS].columns).values
    return data

    

def load_train_test(imputation, sampling, reload=False):
    if exists(TEST_PATH) and not reload:
        return load_data(TEST_PATH, imputation, sampling)

def reset_train_test():
    remove(TEST_PATH)

class DataManager:
    def __init__(self, imputation=None, sampling=None):
        self.imputation = imputation
        self.sampling = sampling
        self.train = load_data(DATA_PATH, imputation, sampling)
        self.test = load_train_test(imputation, sampling)

    @staticmethod
    def reload_data(self):
        self.train = load_data(DATA_PATH, self.imputation, self.sampling)
        self.test = load_train_test(self.imputation, self.sampling)
