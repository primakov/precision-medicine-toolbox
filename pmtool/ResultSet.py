# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:05:22 2022

@author: m.beuque
"""

import numpy as np
import pandas as pd
from warnings import warn


def list_error(temp):
    """Return error if the object format is not list."""
    if type(temp) is list:
        return temp
    else:
        raise TypeError('wrong format, must be list')


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
            train_labels: Labels used for training a model.
            train_predictions: Output predictions on the training dataset of the model between 0 and 1.
            test_labels: Ground truth of the test dataset.
            test_predictions: Output predictions on the testing dataset of the model between 0 and 1.
            external_labels: Ground truth of the external dataset.
            external_predictions: Output predictions on the external dataset of the model between 0 and 1.
        """
        np.random.seed(32)
        self._train_labels = list_error(train_labels)

        self._train_predictions = list_error(train_predictions)
        self.train_df = pd.DataFrame()

        self._test_labels = list_error(test_labels)
        self._test_predictions = list_error(test_predictions)
        self.test_df = pd.DataFrame()

        self._external_labels = list_error(external_labels)
        self._external_predictions = list_error(external_predictions)
        self.external_df = pd.DataFrame()

        self._threshold = -1

        self.__create_dataframes()


    def __create_dataframes(self):

        """Transform the lists of labels and predictions into dataframes."""

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
