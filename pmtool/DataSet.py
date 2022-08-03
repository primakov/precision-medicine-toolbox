# -*- coding: utf-8 -*-
"""
Created on Sun Apr 5 22:40:52 2020
@author: S.Primakov
s.primakov@maastrichtuniversity.nl

"""
import os,re
import pydicom
from warnings import warn
from collections import OrderedDict
from tqdm import tqdm


is_in_list = lambda names_list, name: any([True for v in names_list if re.search(v, name.lower())])

class DataSet:
    '''This class creates imaging dataset objects.'''

    def __init__(self, data_path: str = None,
                 data_type: str = 'dcm', mask_names: list = [],
                 image_only: bool = False, multi_rts_per_pat: bool = False,
                 messy_structure: bool = False,
                 image_names: list=['image', 'volume', 'img', 'vol']):
        """Initialise a dataset object.

        Arguments:
            data_path: Path to DICOM/NRRD root directory (structure inside patient’s folder doesn’t matter).
            data_type: Original data format, 'dcm' is default, can be 'nrrd'.
            mask_names: List of names for NRRD files containing binary mask, default is ‘mask’.
            image_only: If your dataset has only images, whithout Rtstructures, you need to set it to True, otherwise default value is False.
            multi_rts_per_pat: If you have multiple Rtstructures in the patient folder you need to set it to True, otherwise default value is False, to speed up the parsing algorithm.
            messy_structure: If data is not organised properly in folders.
            image_names: List of names for NRRD files containing image file, default is ['image','volume','img','vol'].
        """

        self.index = 0
        self._data_path = data_path
        self._data_type = data_type
        self.__patients = os.listdir(data_path)
        self._patient_dict = OrderedDict() 
        if mask_names:
            self.__mask_names = mask_names
        else:
            self.__mask_names = ['mask']
        
        self._image_only = image_only
        self.__image_names = image_names
        self.__multi_rts_per_pat = multi_rts_per_pat
        self.__parse_directory(messy_structure)
        
        
    def __parse_directory(self, messy_structure=False):
        
        if self._data_type =='nrrd':
            
            for patient in tqdm(self.__patients):
                temp_files = []
                for root, dirs, files in os.walk(os.path.join(self._data_path,patient)):
                    for file in files:
                        if file.endswith('.nrrd'):
                            temp_files.append(os.path.join(root,file))
                if not len(temp_files):
                    raise FileNotFoundError('No nrrd/mha data found: check the folder')
            
                for mask_name in self.__mask_names:         
                    for file in temp_files:
                        if is_in_list(self.__image_names,file.split(os.sep)[-1][:-5]):
                            if not self._image_only:
                                for mfile in temp_files:
                                    if re.search(mask_name.lower(),mfile.lower()):
                                        self._patient_dict[str(patient+'_'+ mfile.split(os.sep)[-1][:-5])]=[file,mfile] #[-20:-5]
                            else:
                                self._patient_dict[str(patient)]=[file]    
            
        elif self._data_type =='dcm':
            for patient in tqdm(self.__patients):
            
                dcm_files=[]
                all_dicom_subfiles = []
                dict_length = len(self._patient_dict)

                if messy_structure:
                    for root, dirs, files in os.walk(os.path.join(self._data_path)):
                        for file in files:
                            if file.endswith('.dcm'):
                                temp_file = pydicom.read_file(os.path.join(root, file),force = True)
                                if temp_file.Modality == 'RTSTRUCT':
                                    all_dicom_subfiles.append(os.path.join(root, file))
                    print(len(all_dicom_subfiles))
                
                for root, dirs, files in os.walk(os.path.join(self._data_path,patient)):
                    for file in files:
                        if file.endswith('.dcm'):
                            dcm_files.append(os.path.join(root, file))
                            
                if not len(dcm_files):
                    warn('No dcm data found for patient:%s check the folder, ensure that dicom files ends with .dcm'%patient)

                if messy_structure:
                    for rt in all_dicom_subfiles:
                        temp_rt = pydicom.read_file(rt, force=True)
                        for dfile in dcm_files:
                            temp_dfile = pydicom.read_file(dfile, force=True)
                            if is_in_list(['ct', 'mr', 'us', 'nm'], str(temp_dfile.Modality).lower()):
                                if (temp_dfile.SeriesInstanceUID != temp_rt.SeriesInstanceUID):
                                    pass
                                else:
                                    datafile = os.path.dirname(os.path.abspath(dfile))
                                    rts_name = rt[:-4].split(os.sep)[-2]  # +rt[:-4].split(os.sep)[-2]
                                    self._patient_dict[patient + '_' + rts_name[-15:]] = [datafile, rt]
                                    break
                else:
                    for file in dcm_files:
                        structfile = ''
                        try:
                            temp_file = pydicom.read_file(file, force=True)

                            if not self._image_only:
                                if temp_file.Modality == 'RTSTRUCT':
                                    structfile = file
                                    datafile = ''
                                    for dfile in dcm_files:
                                        temp_dfile = pydicom.read_file(dfile, force=True)

                                        if is_in_list(['ct', 'mr', 'us', 'nm'], str(temp_dfile.Modality).lower()):
                                            if (temp_dfile.StudyInstanceUID != temp_file.StudyInstanceUID):
                                                warn('StudyInstanceUID doesnt match!')

                                            datafile = os.path.dirname(os.path.abspath(dfile))
                                            rts_name = structfile[:-4].split(os.sep)[-1]
                                            self._patient_dict[patient + '_' + rts_name[-15:]] = [datafile, structfile]
                                            break

                            else:
                                datafile = ''
                                for dfile in dcm_files:
                                    temp_dfile = pydicom.read_file(dfile, force=True)
                                    if is_in_list(['ct', 'mr', 'us', 'nm'], str(temp_dfile.Modality).lower()):
                                        datafile = os.path.dirname(os.path.abspath(dfile))  # getting directory name
                                        self._patient_dict[patient] = [datafile]
                                        break

                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            warn('Some problems have occured with the file: %s' % file)

                        if not self.__multi_rts_per_pat and dict_length - len(self._patient_dict):
                            break
                
      
        else:
            raise NotImplementedError('Currently only "dcm" format and "nrrd" (nrrd/mha) formats are supported')
      
    def __iter__(self):
        return self
    def __len__(self):
        return(len(self._patient_dict))
    
    def __next__(self):
        if self.index == len(self._patient_dict):
            self.index = 0
            raise StopIteration
        temp_key = [*self._patient_dict.keys()][self.index]
        temp_data = self._patient_dict[temp_key]
        self.index +=1
        return temp_key,temp_data
    

