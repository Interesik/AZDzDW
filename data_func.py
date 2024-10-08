import pandas as pd
from os.path import abspath, exists
from os import remove

DATA_PATH = abspath("./mitbih_train_15.csv")
TEST_PATH = abspath("./mitbih_test_head.csv")


def load_data(path):
    data = pd.read_csv(path)
    # skip unamed
    data = data.drop(data.columns[0], axis=1)
    print(data.columns.tolist())
    return data
    

def load_train_test(reload=False):
    if exists(TEST_PATH) and not reload:
        return load_data(TEST_PATH)

def reset_train_test():
    remove(TEST_PATH)

class DataManager:
    def __init__(self):
        self.train = load_data(DATA_PATH)
        self.test = load_train_test()

    @staticmethod
    def reload_data(self):
        self.train = load_data(DATA_PATH)
        self.test = load_train_test()
