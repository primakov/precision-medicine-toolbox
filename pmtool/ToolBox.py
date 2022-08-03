# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:13:55 2020
@author: S.Primakov
s.primakov@maastrichtuniversity.nl
"""

import os,re,logging
from pmtool.DataSet import DataSet
import pydicom
import numpy as np
from skimage import draw
import SimpleITK as sitk
from warnings import warn
from tqdm import tqdm
import matplotlib.pyplot as plt
import radiomics
from radiomics import featureextractor
from pandas import DataFrame
import pandas as pd
import csv
import cv2

class MyException(Exception):
    pass

class ToolBox(DataSet):

    '''This module is inherited from DataSet class and allows for high-level functionality while working with the raw imaging data.'''

    def get_dataset_description(self, parameter_list: list =['Modality', 'SliceThickness',
                    'PixelSpacing', 'SeriesDate', 'Manufacturer']) -> DataFrame:
        """Get specified metadata from the DICOM files of the dataset.

        Arguments:
            parameter_list: List of the parameters to be collected from DICOM files.

        Returns:
            DataFrame with the specified parameters for each patient.
        """
        CT_params = ['PatientName','ConvolutionKernel','SliceThickness',
                      'PixelSpacing','KVP','Exposure','XRayTubeCurrent',
                      'SeriesDate']
        MRI_params = ['Manufacturer','SliceThickness','PixelSpacing',
                      'StudyDate','MagneticFieldStrength','EchoTime']

        if self._data_type =='dcm':
            if parameter_list == 'MRI':
                params_list = MRI_params

            elif parameter_list == 'CT':
                params_list = CT_params
            else:
                params_list = parameter_list

            dataset_stats = DataFrame(data=None, columns = params_list )
            for pat,path in tqdm(self, desc='Patients processed'):
                image,_ = self.__read_scan(path[0])
                for i,temp_slice in enumerate(image):
                    dataset_stats = dataset_stats.append(pd.Series([pat,str(i),*[self.__val_check(temp_slice,x) for x in params_list]],
                                                                   index = ['patient','slice#',*params_list]),
                                                         ignore_index=True)

            return dataset_stats
        else:
            warn('Only available for DICOM dataset')

    def get_jpegs(self, export_path: str):
        '''Convert each slice of nnrd containing ROI to JPEG image. Quick and convienient way to check your ROIs after conversion.

        Arguments:
            export_path: Path to the folder where the JPEGs will be generated.
        '''
        if self._data_type =='nrrd':
            for pat,path in tqdm(self, desc='Patients processed'):
                try:
                    temp_data = sitk.ReadImage(path[0])
                    temp_mask = sitk.ReadImage(path[1])
                    temp_image_array = sitk.GetArrayFromImage(temp_data)
                    temp_mask_array = sitk.GetArrayFromImage(temp_mask)

                    directory = os.path.join(export_path,'images_quick_check',pat,path[1][:-5].split(os.sep)[-1])
                    z_dist = np.sum(temp_mask_array,axis = (1,2))
                    z_ind = np.where(z_dist!=0)[0]

                    for j in z_ind:
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        temp_image_array[j,0,0]= 1
                        temp_mask_array [j,0,0]= 1
                        plt.figure(figsize=(20,20))
                        plt.subplot(121)
                        plt.imshow(temp_image_array[j,...],cmap = 'bone')
                        plt.subplot(122)
                        plt.imshow(temp_image_array[j,...],cmap = 'bone')
                        plt.contour(temp_mask_array[j,...],colors = 'red',linewidths = 2)#,alpha=0.7)
                        plt.savefig(os.path.join(directory,'slice #%d'%j),bbox_inches='tight')
                        plt.close()
                except Exception:
                    warn('Something wrong with %s'%pat)

        else:
            raise TypeError('The toolbox should be initialized with data_type = "nrrd" ')

    def extract_features(self, params_file: str, loggenabled: bool =False) -> DataFrame:
        """Extract PyRadiomic features from the dataset.

                Arguments:
                    params_file: File with the PyRadiomics parameters for features extraction.
                    loggenabled: Enable/disable log file writing.

                Returns:
                    DataFrame with the extracted features.
        """

        if self._data_type =='nrrd':
            #set up pyradiomics
            if loggenabled:
                logger = radiomics.logger
                logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

                # Write out all log entries to a file
                handler = logging.FileHandler(filename='test_log.txt', mode='w')
                formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            # Initialize feature extractor using the settings file
            feat_dictionary,key_number={},0
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

            for pat,path in tqdm(self,desc='Patients processed'):
                try:
                    temp_data = sitk.ReadImage(path[0])
                    temp_mask = sitk.ReadImage(path[1])
                    pat_features = extractor.execute(temp_data, temp_mask)

                    if pat_features['diagnostics_Image-original_Hash'] != '':
                        pat_features.update({'Patient':pat,'ROI':path[1].split(os.sep)[-1][:-5]})
                        feat_dictionary[key_number] = pat_features
                        key_number+=1

                except KeyboardInterrupt:
                    raise
                except Exception:
                    warn('region : %s skipped'%pat)

            output_features = DataFrame.from_dict(feat_dictionary).T

            return output_features
        else:
            raise TypeError('The toolbox should be initialized with data_type = "nrrd"')

    def convert_to_nrrd(self, export_path: str, region_of_interest: str = 'all', image_type=np.int16):
        '''Convert DICOM dataset to the volume (NRRD) format.

        Arguments:
            export_path: Path to the folder where the converted NRRDs will be placed.
            region_of_interest: If you know exact name of the ROI you want to extract, then write it with the ! character in front, eg. region_of_interest = !gtv1 , if you want to extract all the GTVs in the rtstructure eg. gtv1, gtv2, gtv_whatever, then just specify the stem word eg. region_of_interest = gtv, default value is region_of_interest ='all' , which means that all ROIs in rtstruct will be extracted.
            image_type: Data type of the input image.
        '''
        if self._data_type == 'dcm':

            if self._image_only:

                for pat, pat_path in tqdm(self, desc='Patients converted'):
                    img_path = pat_path[0]
                    image = self.__get_image(img_path, image_type)

                    export_dir = os.path.join(export_path, 'converted_nrrds', pat)
                    if not os.path.exists(export_dir):
                        os.makedirs(export_dir)

                    image_file_name = 'image.nrrd'
                    sitk.WriteImage(image, os.path.join(export_dir, image_file_name))

            else:

                for pat, pat_path in tqdm(self, desc='Patients converted'):
                    img_path = pat_path[0]
                    rt_path = pat_path[1]
                    try:
                        rt_structure, roi_list = self.__get_roi(region_of_interest, rt_path)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        roi_list = []
                        warn('Error: ROI extraction failed for patient%s' % pat)

                    for roi in roi_list:
                        try:
                            image, mask = self.__get_binary_mask(img_path, rt_structure, roi, image_type)

                            export_dir = os.path.join(export_path, 'converted_nrrds', pat)

                            if not os.path.exists(export_dir):
                                os.makedirs(export_dir)

                            image_file_name = 'image.nrrd'
                            mask_file_name = '%s_mask.nrrd' % roi
                            sitk.WriteImage(image, os.path.join(export_dir,
                                                                image_file_name))  # save image and binary mask locally
                            sitk.WriteImage(mask, os.path.join(export_dir, mask_file_name))
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            warn('Patients %s ROI : %s skipped' % (pat, roi))


        else:
            raise TypeError('Currently only conversion from dicom -> nrrd is available')

    def pre_process(self, ref_img_path: str = None, save_path: str = None,
                    z_score: bool = False, norm_coeff: tuple = None, hist_match: bool = False,
                    hist_equalize: bool = False, binning: bool = False, percentile_scaling: bool = False,
                    corr_bias_field: bool = False,
                    subcateneus_fat: bool = False, fat_value: bool = None,
                    reshape: bool = False, to_shape: np.array = None,
                    window_filtering_params: tuple = None,
                    verbosity: bool = False, visualize: bool = False):
        '''Pre-process the images.

        Arguments:
            ref_img_path: Path to the reference image for the histogram matching.
            save_path: Path to the folder where the pre-processed images will be stored.
            z_score: Enable Z-scoring.
            norm_coeff: Normalization coefficients for z-scoring (mean and standard deviation).
            hist_match: Enable histogram matching.
            hist_equalize: Enable histogram equalization.
            binning: Enable intensities resampling.
            percentile_scaling: Enable intensities scaling based on 95 percentile.
            corr_bias_field: Enable bias field correction.
            subcateneus_fat: Enable intensities rescaling based on subcateneus fat.
            fat_value: Intensity value specific for the fat.
            reshape: Enable 3D reshaping of the images.
            to_shape: Target shape for image reshaping.
            verbosity: Enable log reporting.
            visualize: Enable visualization of every pre-processing step.
        '''

        ref_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(ref_img_path))
        for i, pat in tqdm(self):
            image = sitk.ReadImage(pat[0])
            mask = sitk.ReadImage(pat[1])
            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)
            ref_image_arr = ref_img_arr.copy()
            # run the pre-processing
            pre_processed_arr = self.__preprocessing_function(image_array, mask_array, ref_image_arr,
                                                              z_score, norm_coeff, hist_match, hist_equalize,
                                                              binning, percentile_scaling,
                                                              corr_bias_field, window_filtering_params,
                                                              subcateneus_fat,
                                                              fat_value,
                                                              reshape, to_shape,
                                                              verbosity, visualize)

            export_dir = os.path.join(save_path, i)
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            pre_processed_image = sitk.GetImageFromArray(pre_processed_arr)
            pre_processed_image.CopyInformation(image)
            sitk.WriteImage(pre_processed_image,
                            os.path.join(export_dir, 'pre_processed_img.nrrd'))  # save image and binary mask locally
            sitk.WriteImage(mask, os.path.join(export_dir, 'mask.nrrd'))

        params = {'ref_img_path': ref_img_path, 'save_path': save_path, 'z_score': z_score, 'norm_coeff': norm_coeff,
                  'hist_match': hist_match, 'hist_equalize': hist_equalize, 'binning': binning,
                  'percentile_scaling': percentile_scaling, 'subcateneus_fat': subcateneus_fat, 'fat_value': fat_value,
                  'verbosity': verbosity, 'visualize': visualize}

        with open(os.path.join(save_path, 'pre-processing_parameters.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Pre-processing parameters:'])
            for key, value in params.items():
                writer.writerow([str(str(key) + ': ' + str(value))])

    def get_quality_checks(self, qc_parameters: dict = {'specific_modality': '',
                                                        'thickness_range': [],
                                                        'scan_length_range': [],
                                                        'axial_res': [],
                                                        'spacing_range': [],
                                                        'kernels_list': []},
                           verbosity: bool =False) -> DataFrame:

        '''Perform a basic quality check for the data. If one of the quality checked parameters is not passed, the correspondig check is not to be performed.

                Arguments:
                    specific_modality str: Target modality of the dataset.
                    thickness_range list: Range of the acceptable slice thicknesses, minimum and maximum.
                    scan_length_range list: Range of the acceptable number of slices, minimum and maximum.
                    axial_res list: Target axial resolution, in both dimensions.
                    spacing_range list: Range of the acceptable in-plane pixel spacing, minimum and maximum.
                    kernels_list list: List of the acceptable kernels for CT data.
                    verbosity: Enable log reporting.
                Returns:
                    DataFrame with the quality check results for each patient. If '1' - check passed, if '0' - check failed.
                '''
        ## init dataframe, parse qa tags
        df_columns = ['Patient id', 'Modality is acceptable', 'Projection is axial',
                      'Complete scan (no missing/overlapping slices)',
                      'Scan len is in range', 'Slice thickness is in range',
                      'Slice thickness is consistent', 'Pixel spacing is in range',
                      'Convolutional kernel tag is present', 'Convolutional kernel is acceptable',
                      'Axial pr. resolution is acceptable', 'Intensity intercept/slope tags are present']

        checks_df = pd.DataFrame([], columns=df_columns)
        for pat in tqdm(self):
            checks = self.__quality_checks(*pat, qc_parameters=qc_parameters, columns=df_columns, verbosity=verbosity)
            checks_df = checks_df.append(checks, ignore_index=True)
        return checks_df

    def __quality_checks(self, patient, path, qc_parameters, columns, verbosity):
        scans, skipped_files = self.__read_scan(path[0])
        scans_sorted = False
        try:
            scans.sort(key=lambda x: x.ImagePositionPatient[2])
            scans_sorted = True
        except Exception:
            warn('Some problems with sorting scans for pat:%s' % patient)
        img = scans[0]

        ##check scans modality
        if qc_parameters.get('specific_modality', 0):
            if len(qc_parameters['specific_modality']) > 0:
                try:
                    modality = self.__check_modality(scans, qc_parameters['specific_modality'])
                    if verbosity:
                        print('Modality check status:', modality)
                except Exception:
                    modality = np.nan
                    print('Cannot perform Modality check for pat: %s' % patient)
            else:
                raise MyException(
                    "Please correctly specify the 'specific_modality' in parameters (e.g. 'specific_modality':'CT')")
        else:
            modality = '-'
            if verbosity:
                print("Acceptable value was not provided for the 'specific_modality', modality check was not performed ")

        ##number of slices_check
        if qc_parameters.get('scan_length_range', 0):
            if len(qc_parameters['scan_length_range']) == 2:
                try:
                    if len(scans) >= qc_parameters['scan_length_range'][0] \
                            and len(scans) <= qc_parameters['scan_length_range'][1]:
                        slice_nr = 1
                    else:
                        slice_nr = 0
                    if verbosity:
                        print('Number of slices check status:', slice_nr)
                except Exception:
                    slice_nr = np.nan
                    print('Cannot perform slice # check for pat: %s' % patient)
            else:
                raise MyException("Please correctly specify the scan_length_range in parameters (e.g. 'scan_length_range':[1,50])")
        else:
            slice_nr = '-'
            if verbosity:
                print('Acceptable range was not provided for the number of slices check.')

        ##pixel_spacing range check
        if qc_parameters.get('spacing_range', 0):
            if len(qc_parameters['spacing_range']) == 2:
                try:
                    ps = img.PixelSpacing
                    if ps[0] == ps[1] and ps[0] >= qc_parameters['spacing_range'][0] and ps[0] <= qc_parameters['spacing_range'][1]:
                        pixel_sp = 1
                    else:
                        pixel_sp = 0
                    if verbosity:
                        print('Pixel spacing in acceptable range check status:', pixel_sp)
                except Exception:
                    pixel_sp = np.nan
                    print('Cannot perform pixel spacing range check for pat: %s' % patient)
            else:
                raise MyException("Please correctly specify the spacing_range in parameters (e.g. 'spacing_range':[0.9,1.5])")
        else:
            pixel_sp = '-'
            if verbosity:
                print('Acceptable range was not provided for the pixel spacing range check.')

        ##check axial
        try:
            axial = self.__check_image_axial_plane(img)
            if verbosity:
                print('Axial plane check status:', axial)
        except Exception:
            axial = np.nan
            print('Cannot perform axial plane check for pat: %s' % patient)

        ##check rescaling constants
        if qc_parameters.get('axial_res', 0):
            if len(qc_parameters['axial_res']) == 2:
                try:
                    intercept_slope, image_shape = self.__get_pixel_values(scans,np.int8,True,qc_parameters['axial_res'])
                    if verbosity:
                        print('Intercept/slope check status:', intercept_slope)
                        print('Image shape is %s check status'%qc_parameters['axial_res'], image_shape)
                except Exception:
                    intercept_slope, image_shape = np.nan, np.nan
                    print('Cannot perform Intercept/slope and image shape check for pat: %s' % patient)
            else:
                raise MyException("Please correctly specify the axial_res in parameters (e.g. 'axial_res':[512,512])")
        else:
            intercept_slope, image_shape = '-', '-'
            if verbosity:
                print('Acceptable range was not provided for the axial resolution, axial pr. resolution check & Intercept/slope check are not performed ')

        ##check slice thickness consistency
        if qc_parameters.get('thickness_range', 0):
            if len(qc_parameters['thickness_range']) == 2:
                try:
                    slice_thickness_consistency, slice_thickness_range = self.__check_consistent_slice_thickness(scans,
                                                                                                              thickness_range=qc_parameters['thickness_range'])
                    if verbosity:
                        print('Slice thickness consistency check status:', slice_thickness_consistency)
                        print('Slice thickness in acceptable range check status:', slice_thickness_range)
                except Exception:
                    slice_thickness_consistency, slice_thickness_range = np.nan, np.nan
                    print('Cannot perform slice thickness consistency and range check for pat: %s' % patient)
            else:
                raise MyException("Please correctly specify the 'thickness_range' in parameters (e.g. 'thickness_range':[3,5])")
        else:
            slice_thickness_consistency, slice_thickness_range = '-', '-'
            if verbosity:
                print('Acceptable range was not provided for the thickness_range, slice thickness range check & slice thickness consistency check are not performed ')

        ##check missing overlaping slices
        try:
            missing_overlapping_slices = self.__check_missing_overlapping_slices(scans)
            if verbosity:
                print('Missing/overlapping slices check status:', missing_overlapping_slices)
        except Exception:
            missing_overlapping_slices = np.nan
            print('Cannot perform missing/overlapping slice check for pat: %s' % patient)

        ##check conv kernel
        if qc_parameters.get('kernels_list', 0):
            if len(qc_parameters['kernels_list']) >0:
                try:
                    conv_kern = self.__check_kernel(scans, qc_parameters['kernels_list'])
                    conv_pres = 1
                    if verbosity:
                        print('Conv Kernel check status:', conv_kern)
                except Exception:
                    conv_kern = np.nan
                    conv_pres = 0
                    print('Cannot perform conv kernel check for pat: %s' % patient)
            else:
                raise MyException("Please correctly specify the 'kernels_list' in parameters (e.g. 'kernels_list':['standard','lung'])")
        else:
            conv_kern, conv_pres = '-', '-'
            if verbosity:
                print("Acceptable value was not provided for the 'kernels_list', conv kernel check was not performed ")

        checks = [patient, modality, axial, missing_overlapping_slices, slice_nr, slice_thickness_range,
                  slice_thickness_consistency, pixel_sp, conv_pres, conv_kern,
                  image_shape, intercept_slope]
        return pd.Series(checks, index=columns)

    def __check_image_axial_plane(self, img):
        if int(img.ImageOrientationPatient[0]) and int(img.ImageOrientationPatient[4]) and not int(
                img.ImageOrientationPatient[1]) and not int(img.ImageOrientationPatient[2]) and not int(
                img.ImageOrientationPatient[3]) and not int(img.ImageOrientationPatient[5]):
            return 1
        else:
            return 0

    def __check_modality(self,scans,speciefic_mod):
        mod_list = [str(x.Modality).lower() for x in scans]
        if mod_list.count(speciefic_mod.lower()) == len(mod_list):
            return 1
        else:
            return 0

    def __check_consistent_slice_thickness(self, scans, thickness_range):
        slice_thickness = [np.round(x.SliceThickness, 1) for x in scans]
        if slice_thickness.count(slice_thickness[0]) == len(slice_thickness):
            if slice_thickness[0] >= thickness_range[0] and slice_thickness[1] <= thickness_range[1]:
                return 1, 1
            else:
                return 1, 0
        else:
            if slice_thickness[0] >= thickness_range[0] and slice_thickness[1] <= thickness_range[1]:
                return 0, 1
            else:
                return 0, 0

    def __check_missing_overlapping_slices(self, scans):
        temp_spacing = []
        scan_real_range = np.round(list(np.float(x.ImagePositionPatient[2]) for x in scans), 1)
        for i in range(len(scan_real_range) - 1):
            temp_spacing.append(np.round(scan_real_range[i + 1] - scan_real_range[i], 1))

        if temp_spacing.count(temp_spacing[0]) == len(temp_spacing):
            return 1
        else:
            return 0

    def __check_kernel(self, scans, kernels=[]):
        kernels = [x.lower() for x in kernels]
        kernels_list = [x.ConvolutionKernel for x in scans]
        if kernels_list.count(kernels_list[0]) == len(kernels_list) and kernels_list[0].lower() in set(kernels):
            return 1
        else:
            return 0

    def __get_roi_id(self,rtstruct,roi):

        for i in range(len(rtstruct.StructureSetROISequence)):
            if str(roi)==rtstruct.StructureSetROISequence[i].ROIName:
                roi_number = rtstruct.StructureSetROISequence[i].ROINumber
                break
        for j in range(len(rtstruct.StructureSetROISequence)):
            if (roi_number==rtstruct.ROIContourSequence[j].ReferencedROINumber):
                break
        return j

    def __get_roi(self,region_of_interest,rt_path):

        rt_structure = pydicom.read_file(rt_path,force=True)
        roi_list = []
        if region_of_interest.lower()=='all':
            roi_list = [rt_structure.StructureSetROISequence[x].ROIName for x in range(0,len(rt_structure.StructureSetROISequence))]
        else:
            for i in range(0,len(rt_structure.StructureSetROISequence)):
                if region_of_interest.lower()[0]=='!':
                    exact_roi = region_of_interest[1:]
                    if exact_roi.lower()== str(rt_structure.StructureSetROISequence[i].ROIName).lower():
                        roi_list.append(rt_structure.StructureSetROISequence[i].ROIName)       ## only roi with the exact same name

                elif re.search(region_of_interest.lower(),str(rt_structure.StructureSetROISequence[i].ROIName).lower()):
                    roi_list.append(rt_structure.StructureSetROISequence[i].ROIName)

        return rt_structure,roi_list

    def __coordinates_to_mask(self,row_coords, col_coords, shape):
        mask = np.zeros(shape)
        pol_row_coords, pol_col_coords = draw.polygon(row_coords, col_coords, shape)
        mask[pol_row_coords, pol_col_coords] = 1
        return mask

    def __read_scan(self, path):
        scan = []
        skiped_files = []
        for s in os.listdir(path):
            try:
                temp_file = pydicom.read_file(os.path.join(path, s), force=True)
                temp_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                temp_mod = temp_file.Modality
                scan.append(temp_file)
                if (temp_mod == 'RTSTRUCT') or (temp_mod == 'RTPLAN') or (temp_mod == 'RTDOSE'):
                    scan.remove(temp_file)
            except Exception:
                skiped_files.append(s)

        try:
            scan.sort(key = lambda x: x.ImagePositionPatient[2])
        except Exception:
            warn('Some problems with sorting scans')

        return scan,skiped_files

    def __get_pixel_values(self,scans,image_type,qa=False,image_res=[512,512]):
        try:
            image = np.stack([s.pixel_array*s.RescaleSlope+s.RescaleIntercept for s in scans])
            if qa:
                if image.shape[1] == image_res[0] and image.shape[2] == image_res[1]:
                    return 1, 1
                else:
                    return 1, 0
            else:
                return image.astype(image_type)
        except Exception:
            warn('Problems occured with rescaling intensities')
            image = np.stack([s.pixel_array for s in scans])
            if qa:
                return 0, 0
            else:
                return image.astype(image_type)

    def __get_image(self, img_path,image_type):

        image, _ = self.__read_scan(img_path)
        img_first_slice = image[0]
        img_array = self.__get_pixel_values(image, image_type)

        xres = np.array(img_first_slice.PixelSpacing[0])
        yres = np.array(img_first_slice.PixelSpacing[1])
        zres = np.abs(image[1].ImagePositionPatient[2] - image[0].ImagePositionPatient[2])
        image_sitk = sitk.GetImageFromArray(img_array.astype(image_type))
        image_sitk.SetSpacing((float(xres), float(yres), float(zres)))
        image_sitk.SetOrigin(
            (float(img_first_slice.ImagePositionPatient[0]), float(img_first_slice.ImagePositionPatient[1]),
             float(img_first_slice.ImagePositionPatient[2])))


        return image_sitk

    def __get_binary_mask(self,img_path,rt_structure,roi,image_type):

        image,_ = self.__read_scan(img_path)

        precision_level = 0.5
        img_first_slice = image[0]
        img_array = self.__get_pixel_values(image,image_type)

        img_length=len(image)

        mask=np.zeros([img_length, img_first_slice.Rows, img_first_slice.Columns],dtype=np.uint8)
        xres=np.array(img_first_slice.PixelSpacing[0])
        yres=np.array(img_first_slice.PixelSpacing[1])
        zres=np.abs(image[1].ImagePositionPatient[2] - image[0].ImagePositionPatient[2])
        roi_id = self.__get_roi_id(rt_structure,roi)

        Fm = np.zeros((3,2))
        Fm[:,0],Fm[:,1] = img_first_slice.ImageOrientationPatient[:3],img_first_slice.ImageOrientationPatient[3:]
        row_pr,column_pr = img_first_slice.PixelSpacing[0],img_first_slice.PixelSpacing[1]

        T_vect = img_first_slice.ImagePositionPatient
        T_1 = np.array(img_first_slice.ImagePositionPatient)
        T_n = np.array(image[-1].ImagePositionPatient)
        k_column = (T_1 - T_n)/(1 - len(image))

        A_multi = np.zeros((4,4))
        A_multi[:3,0],A_multi[:3,1] = Fm[:,0]*row_pr,Fm[:,1]*column_pr
        A_multi[:3,3],A_multi[:3,2] = T_vect,k_column
        A_multi[3,3] = 1

        transform_matrix = np.linalg.inv(A_multi)

        for sequence in rt_structure.ROIContourSequence[roi_id].ContourSequence:
            temp_contour = sequence.ContourData
            contour_first_point = np.array([*sequence.ContourData[:3], 1])
            slice_position = np.dot(transform_matrix,contour_first_point)[2]

            if np.abs(np.round(slice_position)-slice_position)< precision_level:
                assert slice_position < len(image) and slice_position >= 0
                z_index = int(np.round(slice_position))
            else:
                warn('Cant find the slice position, try to select lower precison level')
                warn('Slice position is: ',slice_position,' suggested position is: ',np.round(slice_position))
                z_index = None

            x,y,z=[],[],[]

            for i in range(0,len(temp_contour),3):
                x.append(temp_contour[i+0])
                y.append(temp_contour[i+1])
                z.append(temp_contour[i+2])

            x,y,z=np.array(x),np.array(y),np.array(z)

            coord = np.vstack((x,y,z,np.ones_like(x)))

            pixel_coords_c_r_s = np.dot(transform_matrix,coord)
            img_slice = self.__coordinates_to_mask(pixel_coords_c_r_s[1,:],pixel_coords_c_r_s[0,:],[img_array.shape[1],img_array.shape[2]])

            assert z_index

            mask[z_index,:,:] = np.logical_or(mask[z_index,:,:],img_slice)

        image_sitk = sitk.GetImageFromArray(img_array.astype(image_type))
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.int8))

        image_sitk.SetSpacing((float(xres),float(yres),float(zres)))
        mask_sitk.SetSpacing((float(xres),float(yres),float(zres)))
        image_sitk.SetOrigin((float(img_first_slice.ImagePositionPatient[0]),float(img_first_slice.ImagePositionPatient[1]),
                       float(img_first_slice.ImagePositionPatient[2])))
        mask_sitk.SetOrigin((float(img_first_slice.ImagePositionPatient[0]),float(img_first_slice.ImagePositionPatient[1]),
                       float(img_first_slice.ImagePositionPatient[2])))

        return image_sitk, mask_sitk

    def __val_check(self,file,value):
        try:
            val = getattr(file,value)
            if val == '' or val==' ':
                return 'NaN'
            else:
                return val
        except Exception:
            return 'NaN'

    def __normalize_image_zscore(self, image, norm_coeff):  ##Zscore whole image
        img = image - norm_coeff[0]
        img = img / norm_coeff[1]
        return img

    def __normalize_image_zscore_per_image(self, image, mask, verbosity):  ##Zscore based on the masked region intensities
        image_s = image.copy()
        mu = np.mean(image_s.flatten())
        sigma = np.std(image_s.flatten())
        if verbosity:
            print('ROI MU = %s, sigma = %s' % (mu, sigma))
        img = (image - mu) / sigma
        return img, mu, sigma

    def __resample_intensities(self, orig_img, bin_nr, verbosity):  ## Intensity resampling whole image/or masked region
        v_count = 0
        img_list = []
        filtered = orig_img.copy()
        if np.min(orig_img.flatten()) < 0:
            filtered += np.abs(np.min(orig_img.flatten()))
        resampled = np.zeros_like(filtered)
        max_val_img = np.max(filtered.flatten())
        step = max_val_img / (1.0*bin_nr)
        print('min: ',np.min(filtered.flatten()),'max: ',np.max(filtered.flatten()),'step: ',step)
        for st in np.arange(step, max_val_img + step, step):
            resampled[(filtered <= st) & (filtered >= st - step)] = v_count
            v_count += 1
        if verbosity:
            print("Resampling with a step of: ", step, 'Amount of unique values, original img: ',
                  len(np.unique(orig_img.flatten())), 'resampled img: ', len(np.unique(resampled.flatten())))

        if len(np.unique(resampled.flatten())) < 255:
            return np.array(resampled, dtype=np.uint8)
        else:
            return np.array(resampled, dtype=np.uint16)

    def __intensity_scaling(self, img, fat_int_value=None, method=''):
        if method == 'fat' and fat_int_value:
            filtered = img.copy()
            if np.min(img.flatten()) < 0:
                filtered += np.min(img.flatten())
            filtered = (1.0 * filtered) / (1.0 * np.max(img.flatten()))
            filtered = 1.0 * fat_int_value * filtered
            return np.array(filtered, np.uint16)

        elif method == '95th':
            filtered = img.copy()
            if np.min(img.flatten()) < 0:
                filtered += np.min(img.flatten())
            perc_95 = np.percentile(filtered.flatten(), 95)
            filtered = (1.0 * filtered) / (1.0 * np.max(img.flatten()))
            filtered = 1.0 * perc_95 * filtered
            return np.array(filtered, np.uint16)
        else:
            print('max value is not understood, skipping intensity rescaling step!')

    def __histogram_matching(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape).astype(np.int16)

    def __resize_3d_img(self, img, shape, interp=cv2.INTER_CUBIC):

        init_img = img.copy()
        temp_img = np.zeros((init_img.shape[0], shape[1], shape[2]))
        new_img = np.zeros(shape)
        for i in range(0, init_img.shape[0]):
            temp_img[i, ...] = cv2.resize(init_img[i, ...], dsize=(shape[2], shape[1]), interpolation=interp)
        for j in range(0, shape[1]):
            new_img[:, j, :] = (cv2.resize(temp_img[:, j, :], dsize=(shape[2], shape[0]), interpolation=interp))
        return new_img

    def __correct_bias_field(self, img, mask, n_fitting_levels, n_iterations):

        img_sitk = sitk.GetImageFromArray(img)
        mask_sitk = sitk.GetImageFromArray(mask)

        img_sitk_cast = sitk.Cast(img_sitk, sitk.sitkFloat32)
        mask_sitk_cast = sitk.Cast(mask_sitk, sitk.sitkUInt8)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(n_fitting_levels * n_iterations)

        img_output = sitk.GetArrayFromImage(corrector.Execute(img_sitk_cast, mask_sitk_cast))

        return img_output

    def __apply_window(self, img, window_params, verbosity=False):
        new_img = img.copy()
        WW = window_params[0]
        WL = window_params[1]
        assert (WW < 2000) and (WW > 0) and (WL < 200) and (WL > -1000)
        up_lim, low_lim = WL + WW / 2, WL - WW / 2
        if low_lim < -1000:
            low_lim = -1000
        if verbosity:
            print('Window limits ', low_lim, up_lim)
        new_img[np.where(img < low_lim)] = low_lim
        new_img[np.where(img > up_lim)] = up_lim
        return new_img

    def __preprocessing_function(self, img, mask, ref_img, z_score, norm_coeff, hist_match, hist_equalize, binning,
                                 percentile_scaling, corr_bias_field, window_filtering_params,
                                 subcateneus_fat, fat_value, reshape, to_shape,
                                 verbosity, visualize):

        if verbosity:
            unique_number_of_intensity = np.unique(img.flatten())
            min_int, max_int = np.min(img.flatten()), np.max(img.flatten())
            img_type = img.dtype
            print('-' * 40)
            print('Original image stats:')
            print('Number of unique intensity values :', len(unique_number_of_intensity))
            print('Intensity values range:[%s:%s]' % (min_int, max_int))
            print('Image intensity dtype:', img_type)
            print('-' * 40)
        if visualize:
            plt.figure(figsize=(12, 12))
            plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
            plt.title('original image')
            plt.show()

        if corr_bias_field:
            img = self.__correct_bias_field(img, mask, n_fitting_levels=4, n_iterations=[50])
            if verbosity:
                print('N4 bias field correction performed')
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Bias field correction')
                plt.show()

        if len(window_filtering_params) == 2:
            img = self.__apply_window(img, window_filtering_params,verbosity)
            if verbosity:
                print('Window filtering applied')
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Window filtering')
                plt.show()

        if subcateneus_fat:
            img = self.__intensity_scaling(np.squeeze(img), fat_value, method='fat')
            if verbosity:
                print('Intensity values rescaled based on the subcateneus fat')
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('subcateneus fat scaling')
                plt.show()

        if percentile_scaling:
            img = self.__intensity_scaling(np.squeeze(img), method='95th')
            if verbosity:
                print('Intensity values rescaled based on the 95th percentile')
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('95th percentile scaling')
                plt.show()

        if hist_match:
            img = self.__histogram_matching(np.squeeze(img), np.squeeze(ref_img))
            if verbosity:
                print('Histogram matching was applied')
            if visualize:
                plt.figure(figsize=(22, 12))
                plt.subplot(121)
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Histogram matched')
                plt.subplot(122)
                plt.imshow(ref_img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Ref image')
                plt.show()

        if binning:
            img = self.__resample_intensities(img, binning, verbosity)
            if verbosity:
                print('-' * 40)
                print('Intensity values resampled with a number of bins: %s' % binning)
                print('-' * 40)
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('%s bins resampled' % binning)
                plt.show()

        if hist_equalize and img.dtype == np.uint8:
            equalized = [cv2.equalizeHist(np.squeeze(x).astype(np.uint8)) for x in img]
            img = np.stack(equalized)
            if verbosity:
                print('Histogram equalization applied')
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Histogram equalize')
                plt.show()

        if z_score and norm_coeff:
            img = self.__normalize_image_zscore(img, norm_coeff)
            if verbosity:
                print('Z-score normalization applied based on the whole image, Mu=%s, sigma=%s' % (norm_coeff))
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Z-score normalization applied')
                plt.show()

        elif z_score:
            img, mu, sigma = self.__normalize_image_zscore_per_image(img, verbosity)
            if verbosity:
                print('Z-score normalization applied based on image intensities, Mu=%s, sigma=%s' % (mu, sigma))
            if visualize:
                plt.figure(figsize=(12, 12))
                plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                plt.title('Z-score normalization applied')
                plt.show()

        if reshape and to_shape:
            if to_shape.shape==3:
                img = self.__resize_3d_img(img)
                if verbosity:
                    print('Image reshaped: %s' % to_shape)
                if visualize:
                    plt.figure(figsize=(12, 12))
                    plt.imshow(img[int(len(img) / 2.), ...], cmap='bone')
                    plt.title('Reshaped image')
                    plt.show()

        return img


