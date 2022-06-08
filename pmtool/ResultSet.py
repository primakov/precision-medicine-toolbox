# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:05:22 2022

@author: m.beuque
"""

##could also consider loading a model and predicting on the pre-processed features datasets

import numpy as np
import pandas as pd
from warnings import warn


class ResultSet:
    ''' This class creates result set objects.'''

    def __init__(self,
                 train_labels: list = [],
                 train_predictions: list = [],
                 test_labels: list = [],
                 test_predictions: list = [],
                 external_labels: list = [],
                 external_predictions: list = []
                 ):

        """Initialise a result object.
        Arguments:
            train_labels: labels used for training a model
            train_predictions: output predictions on the training dataset of the model between 0 and 1
            test_labels: ground truth of the test dataset
            test_predictions: output predictions on the testing dataset of the model between 0 and 1
            external_labels:ground truth of the external dataset
            external_predictions:output predictions on the external dataset of the model between 0 and 1
        """
        np.random.seed(32)  # not sure if it is inherited

        self._train_labels = train_labels
        self._train_predictions = train_predictions
        self.train_df = pd.DataFrame()

        self._test_labels = test_labels
        self._test_predictions = test_predictions
        self.test_df = pd.DataFrame()

        self._external_labels = external_labels
        self._external_predictions = external_predictions
        self.external_df = pd.DataFrame()

        self._threshold = -1

        self.__create_dataframes()

    def __create_dataframes(self):

        # transform the lists of labels and predictions into dataframes

        if len(self._train_labels) and len(self._train_labels) == len(self._train_predictions):
            self.train_df = pd.DataFrame({"labels": self._train_labels, "predictions": self._train_predictions})
        elif not len(self._train_labels):
            warn("training dataset not passed")
        elif len(self._train_labels) != len(self._train_predictions):
            warn("length of train labels and predictions don't match")
        if len(self._test_labels) and len(self._test_labels) == len(self._test_predictions):
            self.test_df = pd.DataFrame({"labels": self._test_labels, "predictions": self._test_predictions})
        elif len(self._test_labels) != len(self._test_predictions):
            warn("length of test labels and predictions don't match")
        if len(self._external_labels) and len(self._external_labels) == len(self._external_predictions):
            self.external_df = pd.DataFrame(
                {"labels": self._external_labels, "predictions": self._external_predictions})
        elif len(self._external_labels) != len(self._external_predictions):
            warn("length of external labels and predictions don't match")

## testing class

#test = ResultSet(test_labels = [0,1],test_predictions =[0.1,0.2])