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

# imaging methods tests
def test_get_dataset_description():
    data_dcms = pickle.load(open('data/test/data_dcms.pickle', "rb"))
    assert len(data_dcms.get_dataset_description()) > 0

def test_get_quality_checks():
    qc_params = {'specific_modality': 'ct',
                 'thickness_range': [2, 5],
                 'spacing_range': [0.5, 1.25],
                 'scan_length_range': [5, 170],
                 'axial_res': [512, 512],
                 'kernels_list': ['standard', 'lung', 'b19f']}
    data_dcms = pickle.load(open('data/test/data_dcms.pickle', "rb"))
    assert len(data_dcms.get_quality_checks(qc_params)) > 0

def test_convert_to_nrrd():

    flag_files_created = False

    export_path = 'data/test/'
    nrrd_path = export_path + 'converted_nrrds/'

    data_dcms = pickle.load(open('data/test/data_dcms.pickle', "rb"))
    data_dcms.convert_to_nrrd(export_path, 'gtv')

    flag_folders_created = len(os.listdir(nrrd_path)) > 0

    if flag_folders_created:
        first_folder = os.listdir(nrrd_path)[0]
        first_folder_path = nrrd_path + first_folder
        flag_files_created = len(os.listdir(first_folder_path)) > 0

    data_nrrd = ToolBox(data_path = 'data/test/converted_nrrds/', data_type='nrrd')
    #pickle.dump(data_nrrd, file=open("data/test/data_nrrd.pickle", "wb"))

    assert (flag_folders_created)&(flag_files_created)

def test_get_jpegs():

    flag_files_created = False

    export_path = 'data/test/'
    jpeg_path = export_path + 'images_quick_check/'

    data_nrrd = pickle.load(open("data/test/data_nrrd.pickle", "rb"))
    data_nrrd.get_jpegs(export_path)

    flag_folders_created = len(os.listdir(jpeg_path)) > 0

    if flag_folders_created:
        first_folder = os.listdir(jpeg_path)[0]
        first_folder_path = jpeg_path + first_folder

        second_level_folder = os.listdir(first_folder_path)[0]
        second_level_folder_path = first_folder_path + '/' + second_level_folder

        flag_files_created = len(os.listdir(second_level_folder_path)) > 0

    assert (flag_folders_created)&(flag_files_created)

def test_pre_process():

    flag_files_created = False
    first_folder_path = ''

    export_path = 'data/test/'
    proc_path = export_path + 'nrrd_preprocessed/'

    data_nrrd = pickle.load(open("data/test/data_nrrd.pickle", "rb"))
    data_nrrd.pre_process(ref_img_path='data/test/converted_nrrds_test/sub-001_2/image.nrrd',
                          save_path=proc_path,
                          hist_match=False,
                          subcateneus_fat=False,
                          fat_value=774,
                          percentile_scaling=False,
                          window_filtering_params=(1500, -600),
                          binning=255,
                          verbosity=True,
                          z_score=True,
                          hist_equalize=True,
                          norm_coeff=(1000., 500.),
                          visualize=False)

    flag_folders_created = len(os.listdir(proc_path)) > 0

    if flag_folders_created:

        for item in os.listdir(proc_path):
            candidate_path = proc_path + item
            if os.path.isdir(candidate_path):
                first_folder_path = candidate_path

        flag_files_created = len(os.listdir(first_folder_path)) > 0

    assert (flag_folders_created) & (flag_files_created)

def test_extract_features():

    data_nrrd = pickle.load(open("data/test/data_nrrd.pickle", "rb"))
    parameters = 'examples/example_ct_parameters.yaml'
    features = data_nrrd.extract_features(parameters, loggenabled=True)

    assert len(features) > 0

def test_clean_imaging_module():
    try:
        shutil.rmtree('data/test/converted_nrrds/')
    except:
        print ('NRRD conversion failed.')

    try:
        shutil.rmtree('data/test/images_quick_check/')
    except:
        print ('ROIs check failed.')

    try:
        shutil.rmtree('data/test/nrrd_preprocessed/')
    except:
        print ('Pre-processing failed.')

#features methods tests
def test_plot_distribution():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.plot_distribution(fs._feature_column[:10])

    assert os.path.isfile('data/test/extracted_features_full_distr.html')

def test_plot_correlation_matrix():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.plot_correlation_matrix(fs._feature_column[:10])

    assert os.path.isfile('data/test/extracted_features_full_corr.html')

def test_plot_MW_p():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.plot_MW_p(fs._feature_column[:10])

    assert os.path.isfile('data/test/extracted_features_full_MW.html')

def test_plot_univariate_roc():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.plot_univariate_roc(fs._feature_column[:10])

    assert os.path.isfile('data/test/extracted_features_full_roc-univar.html')

def test_volume_analysis():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.volume_analysis(volume_feature='original_shape_VoxelVolume')

    assert os.path.isfile('data/test/extracted_features_full_volume_corr.html')&os.path.isfile('data/test/extracted_features_full_volume_PRC.html')

def test_calculate_basic_stats():

    fs = pickle.load(open('data/test/fs.pickle', "rb"))
    fs.calculate_basic_stats(volume_feature='original_shape_VoxelVolume')

    assert os.path.isfile('data/test/extracted_features_full_basic_stats.xlsx')

def test_clean_features_module():

    files_to_delete = ['data/test/extracted_features_full_distr.html',
                       'data/test/extracted_features_full_corr.html',
                       'data/test/extracted_features_full_MW.html',
                       'data/test/extracted_features_full_roc-univar.html',
                       'data/test/extracted_features_full_volume_corr.html',
                       'data/test/extracted_features_full_volume_PRC.html',
                       'data/test/extracted_features_full_basic_stats.xlsx'
                       ]

    for filename in files_to_delete:

        try:
            os.remove(filename)
        except:
            print ('Could not remove ', filename)

# results methods tests
def test_get_results():

    result_generation = pickle.load(open('data/test/result_generation.pickle', "rb"))

    result_generation.get_results('train')
    result_generation.get_results('test')
    result_generation.get_results('external')

def test_get_stats_with_ci():

    result_generation = pickle.load(open('data/test/result_generation.pickle', "rb"))

    result_generation.get_stats_with_ci('train')
    result_generation.get_stats_with_ci('test')
    result_generation.get_stats_with_ci('external')

def test_print_confusion_matrix():

    result_generation = pickle.load(open('data/test/result_generation.pickle', "rb"))

    result_generation.print_confusion_matrix('train', ['0', '1'])
    result_generation.print_confusion_matrix('test', ['0', '1'])
    result_generation.print_confusion_matrix('external', ['0', '1'])

def test_plot_roc_auc_ci():

    result_generation = pickle.load(open('data/test/result_generation.pickle', "rb"))
    result_generation.plot_roc_auc_ci(title ="testing roc curve function")












