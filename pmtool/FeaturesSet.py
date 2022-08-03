# -*- coding: utf-8 -*-
"""
Created on Tue Jun 6 13:52:52 2020
@author: E.Lavrova
e.lavrova@maastrichtuniversity.nl
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
import plotly.subplots
from statsmodels.stats import multitest
import os
import scipy as sp
import math
import scipy.stats
from operator import le, gt
from sklearn.metrics import auc
import plotly.io as pio


class FeaturesSet:
    '''This class creates features set objects.'''

    def __init__(self,
                 feature_path: str =None,
                 outcome_path: str =None,
                 feature_column: list =[],
                 feature_column_to_drop: list =[],
                 outcome_column: str ='',
                 patient_column: str ='',
                 patient_in_outcome_column: str ='',
                 patient_to_drop: list =[]):

        """Initialise a dataset object.

        Arguments:
            feature_path: Path to csv./.xls(x) file with features.
            outcome_path: Path to csv./.xls(x) file with outcomes.
            feature_column: List of features to be included.
            feature_column_to_drop: List of features to be excluded.
            outcome_column: Name of the column with outcomes.
            patient_column: Name of the column with patients IDs in features file.
            patient_in_outcome_column: Name of the column with patients IDs in outcomes file.
            patient_to_drop: List of patients to be excluded.
        """

        if type(feature_path) is str:
            self._feature_path = feature_path
        else:
            print('Features csv/xls file path has wrong format.')
        if type(patient_column) is str:
            self._patient_column = patient_column
        else:
            print('Patient column name has wrong format.')
        if type(outcome_path) is str:
            self._outcome_path = outcome_path
        else:
            print('Outcome csv/xls file path has wrong format.')
        if type(feature_column) is list:
            self._feature_column = feature_column
        else:
            print('List of feature columns has wrong format.')
        if type(outcome_column) is str:
            self._outcome_column = outcome_column
        else:
            print('Outcome column name has wrong format.')
        if type(patient_in_outcome_column) is str:
            self._patient_in_outcome_column = patient_in_outcome_column
        else:
            print('Patient column name in dataframe with outcomes has wrong format.')
        if type(patient_to_drop) is list:
            self._patient_to_drop = patient_to_drop
        else:
            print('List of patient names to be excluded has wrong format.')
        if type(feature_column_to_drop) is list:
            self._feature_column_to_drop = feature_column_to_drop
        else:
            print('List of feature names to be excluded has wrong format.')
        self._class_label = []
        self._outcome = []
        self._feature_dataframe = None
        self._feature_outcome_dataframe = None
        if len(feature_path)>0 and len(patient_column)>0:
            self.__read_files()
        else:
            print('Path to csv/xls with features or patient column name is missing.')

    def __read_files(self):

        # reads .csv/.xls(x) tables with features and outcomes and gets feature_set attributes

        if '.csv' in self._feature_path:
            feature_df = pd.read_csv(self._feature_path, dtype={self._patient_column: str})
        elif '.xls' in self._feature_path:
            feature_df = pd.read_excel(self._feature_path, dtype={self._patient_column: str})
        else:
            print('Data format is not supported')
            return

        if len(self._feature_column) > 0:
            self._feature_column = list(set(self._feature_column) & set(list(feature_df.columns)))
        else:
            self._feature_column = list(feature_df.columns)

        if len(self._outcome_column) > 0:
            if self._outcome_column in self._feature_column:
                self._feature_column.remove(self._outcome_column)

        if len(self._feature_column_to_drop) > 0:
            for feature in self._feature_column_to_drop:
                if feature in self._feature_column:
                    self._feature_column.remove(feature)

        if '' in self._feature_column:
            self._feature_column.remove('')
        if 'Unnamed: 0' in self._feature_column:
            self._feature_column.remove('Unnamed: 0')

        technical_features_to_remove = []
        for feature in self._feature_column:
            if ('diagnostics' in feature) or ('general' in feature):
                technical_features_to_remove.append(feature)
        for feature in technical_features_to_remove:
            self._feature_column.remove(feature)

        if len(self._patient_column) > 0:
            if self._patient_column in list(feature_df.columns):
                self._patient_name = list(feature_df[self._patient_column])
                feature_df.set_index(self._patient_column, inplace=True)
                if len(self._patient_to_drop) > 0:
                    for patient in self._patient_to_drop:
                        if patient in self._patient_name:
                            self._patient_name.remove(patient)
                feature_df = feature_df.reindex(self._patient_name)
            if self._patient_column in self._feature_column:
                self._feature_column.remove(self._patient_column)

        self._feature_dataframe = feature_df[self._feature_column].copy()

        if len(self._outcome_path) > 0:
            if (len(self._patient_column) > 0) & (len(self._patient_in_outcome_column) > 0):
                if '.csv' in self._outcome_path:
                    outcome_df = pd.read_csv(self._outcome_path, dtype={self._patient_in_outcome_column: str,
                                                                        self._outcome_column: str})
                elif '.xls' in self._outcome_path:
                    outcome_df = pd.read_excel(self._outcome_path, dtype={self._patient_in_outcome_column: str,
                                                                          self._outcome_column: str})
                outcome_df.set_index(self._patient_in_outcome_column, inplace=True)
                self._outcome = outcome_df[self._outcome_column]

        else:
            if self._outcome_column in list(feature_df.columns):
                self._outcome = feature_df[self._outcome_column]

        if len(self._outcome) > 0:
            self._feature_outcome_dataframe = self._feature_dataframe.copy()
            self._feature_outcome_dataframe[self._outcome_column] = None
            for patient_outcome in list(self._outcome.index):
                for patient in self._patient_name:
                    if patient_outcome in patient:
                        self._feature_outcome_dataframe.at[patient, self._outcome_column] = \
                            self._outcome[patient_outcome]
            self._outcome = self._feature_outcome_dataframe[self._outcome_column]
            self._class_label = pd.unique(np.array(list(self._feature_outcome_dataframe[self._outcome_column])))
            self._class_label.sort()
            data_balance = []
            for label_name in self._class_label:
                data_balance.append(np.sum(np.array(list(self._outcome)) == label_name)/len(self._outcome))

        print('Number of observations: {}\nClass labels: {}\nClasses balance: {}'.format(len(self._outcome),
                                                                                  self._class_label,
                                                                                  data_balance))
        return None












