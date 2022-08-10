"""Unit tests for precision-medicine-toolbox."""

import os, sys
import numpy as np

from pmtool.ToolBox import ToolBox
from pmtool.AnalysisBox import AnalysisBox
from pmtool.GenerateResultBox import GenerateResultBox

import pytest
import pickle

import glob
import shutil

# initialising objects
def test_toolbox_init():
    parameters = {'data_path': 'data/test/dicom_test/',
                  'data_type': 'dcm',
                  'multi_rts_per_pat': False}
    data_dcms = ToolBox(**parameters)

    #pickle.dump(data_dcms, file=open("data/test/data_dcms.pickle", "wb"))

    assert len(data_dcms) > 0

def test_analysisbox_init():
    parameters = {
        'feature_path': 'data/test/extracted_features_full.xlsx',  # path to csv/xls file with features
        'outcome_path': 'data/test/extended_clinical_df.xlsx',  # path to csv/xls file with outcome
        'patient_column': 'Patient',  # name of column with patient ID
        'patient_in_outcome_column': 'PatientID',  # name of column with patient ID in clinical data file
        'outcome_column': '1yearsurvival'  # name of outcome column
    }
    fs = AnalysisBox(**parameters)

    #pickle.dump(fs, file=open("data/test/fs.pickle", "wb"))

    assert len(fs._feature_outcome_dataframe) > 0

def test_generateresultbox_init():

    train_labels = [int(np.round(np.random.uniform(low=0, high=1))) for i in range(100)]
    train_predictions = [np.random.uniform(low=0, high=1) for i in range(100)]
    test_labels = [int(np.round(np.random.uniform(low=0, high=1))) for i in range(50)]
    test_predictions = [np.random.uniform(low=0, high=1) for i in range(50)]
    external_labels = [int(np.round(np.random.uniform(low=0, high=1))) for i in range(50)]
    external_predictions = [np.random.uniform(low=0, high=1) for i in range(50)]

    result_generation = GenerateResultBox(train_labels=train_labels,
                                          train_predictions=train_predictions,
                                          test_labels=test_labels,
                                          test_predictions=test_predictions,
                                          external_labels=external_labels,
                                          external_predictions=external_predictions)

    train_labels_present = len(result_generation._train_labels)>0
    train_predictors_present = len(result_generation._train_predictions) > 0
    test_labels_present = len(result_generation._test_labels) > 0
    test_predictors_present = len(result_generation._test_predictions) > 0
    external_labels_present = len(result_generation._external_labels) > 0
    external_predictors_present = len(result_generation._external_predictions) > 0

    #pickle.dump(result_generation, file=open("data/test/result_generation.pickle", "wb"))

    assert ((train_labels_present)&(train_predictors_present))|((test_labels_present)&(test_predictors_present))|((external_labels_present)&(external_predictors_present))













