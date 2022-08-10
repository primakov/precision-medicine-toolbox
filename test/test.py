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


def test_analysisbox_init():
    parameters = {
        'feature_path': '../data/test/extracted_features_full.xlsx',  # path to csv/xls file with features
        'outcome_path': '../data/test/extended_clinical_df.xlsx',  # path to csv/xls file with outcome
        'patient_column': 'Patient',  # name of column with patient ID
        'patient_in_outcome_column': 'PatientID',  # name of column with patient ID in clinical data file
        'outcome_column': '1yearsurvival'  # name of outcome column
    }
    fs = AnalysisBox(**parameters)
    assert len(fs._feature_outcome_dataframe) > 0

def test_plot_distribution():

    












