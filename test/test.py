"""Unit tests for precision-medicine-toolbox."""

import os, sys
import numpy as np
from pmtool.ToolBox import ToolBox
from pmtool.AnalysisBox import AnalysisBox
from pmtool.GenerateResultBox import GenerateResultBox

import pytest

def test_toolbox_init():
    parameters = {'data_path': r'../data/dcms/',
                  'data_type': 'dcm',
                  'multi_rts_per_pat': False}
    data_dcms = ToolBox(**parameters)
    assert len(data_dcms) > 0

def test_analysisbox_init():
    parameters = {
        'feature_path': "../data/features/extracted_features_full.xlsx",  # path to csv/xls file with features
        'outcome_path': "../data/features/extended_clinical_df.xlsx",  # path to csv/xls file with outcome
        'patient_column': 'Patient',  # name of column with patient ID
        'patient_in_outcome_column': 'PatientID',  # name of column with patient ID in clinical data file
        'outcome_column': '1yearsurvival'  # name of outcome column
    }
    fs = AnalysisBox(**parameters)
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

    assert len(result_generation)>0


