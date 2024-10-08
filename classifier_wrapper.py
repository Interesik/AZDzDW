import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from os.path import abspath
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from imblearn.metrics import specificity_score
from consts import *
from data_func import DataManager

class ClassifierWrapper:

    def __init__(self, classifier, dataManager = None, class_col = CLASS_COL, value_cols = VALUE_COLS, name = None):
        self.classifier = classifier
        self.class_col = class_col
        self.value_cols = value_cols
        self.dataManager = dataManager
        if dataManager is None:
            self.dataManager = DataManager()
        self.name = name if name else classifier.__class__.__name__
        self.label_binarizer = LabelBinarizer()

    def fit(self):
        self.classifier.fit(self.dataManager.train[self.value_cols], self.dataManager.train[self.class_col])

    def fit_labels(self):
        self.label_binarizer.fit(self.dataManager.train[self.class_col])

    def ensure_fitted(func):
        def inner(self, *args, **kwargs):
            try:
                check_is_fitted(self.classifier)
            except NotFittedError:
                self.fit()
            return func(self, *args, **kwargs)
        return inner

    def ensure_labels_fitted(func):
        def inner(self, *args, **kwargs):
            try:
                check_is_fitted(self.label_binarizer)
            except NotFittedError:
                self.fit_labels()
            return func(self, *args, **kwargs)
        return inner
        
    @ensure_fitted
    def predict(self):
        return self.classifier.predict(self.dataManager.test[self.value_cols])

    @ensure_fitted
    def predict_proba(self):
        return self.classifier.predict_proba(self.dataManager.test[self.value_cols])

    @ensure_labels_fitted
    def transform_labels(self):
        return self.label_binarizer.transform(self.dataManager.test[self.class_col])

    def plot_roc(self, figsize=(16,4), file_to_save=None):
        oneVsRestLabels = self.transform_labels()
        fig, axes = plt.subplots(1, len(CLASSES), figsize=figsize)
        
        for c, ax in zip(CLASSES, axes):
            RocCurveDisplay.from_predictions(oneVsRestLabels[:, int(c)], self.predict_proba()[:, int(c)], ax=ax)
            ax.set(title=f"ROC curve for {c} vs Rest")
            
        fig.suptitle(f"ROC curves for {self.name}", verticalalignment="bottom")
        plt.show()

        if file_to_save:
            plt.savefig(abspath(file_to_save))

    def plot_confusion_matrix(self, figsize=(12,6), file_to_save=None):
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay.from_predictions(self.dataManager.test[CLASS_COL], self.predict(), ax=ax)
        ax.set(title=f"Confusion Matrix for {self.name}")
        plt.show()

        if file_to_save:
            plt.savefig(abspath(file_to_save))

    def plot_learning_curve(self, figsize=(12,6), file_to_save=None):
        fig, ax = plt.subplots(figsize=figsize)
        LearningCurveDisplay.from_estimator(self.classifier, self.dataManager.train[self.value_cols], self.dataManager.train[self.class_col], ax=ax)
        ax.set(title=f"Learning curve for {self.name}")
        plt.show()

        if file_to_save:
            plt.savefig(abspath(file_to_save))

    def calc_params(self, average="macro"):
        true = self.dataManager.test[self.class_col]
        pred = self.predict()
        return pd.DataFrame([[
            accuracy_score(true, pred),
            precision_score(true, pred, average=average),
            recall_score(true, pred, average=average),
            specificity_score(true, pred, average=average),
        ]], columns=["accuracy", "precision", "sensitivity","specificity"], index=[self.name])

