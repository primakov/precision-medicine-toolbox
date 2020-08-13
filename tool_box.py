# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:13:55 2020
@author: S.Primakov
s.primakov@maastrichtuniversity.nl
"""

import os,re,logging
from data_set import data_set
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

class tool_box(data_set):

    def get_dataset_description(self,parameter_list=['Modality','SliceThickness',
                    'PixelSpacing','SeriesDate','Manufacturer']):
        '''get dicom dataset parameters'''
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
                           
            dataset_stats = DataFrame(data=None,columns = params_list )
            for pat,path in tqdm(self,desc='Patients processed'):
                image,_ = self.__read_scan(path[0])
                for i,temp_slice in enumerate(image):
                    dataset_stats = dataset_stats.append(pd.Series([pat,str(i),*[self.__val_check(temp_slice,x) for x in params_list]],index = ['patient','slice#',*params_list]),ignore_index=True) 
        
            return dataset_stats
        else:
            warn('Only available for DICOM dataset')

    def get_jpegs(self,export_path):
        '''Convert each slice of nnrd containing ROI to jpeg image. Quick and convienient way to check your ROI's after conversion.
        export_path = .. path to the folder where the jpgs will be generated
        '''
        if self._data_type =='nrrd':
            for pat,path in tqdm(self,desc='Patients processed'):
                try:
                    temp_data = sitk.ReadImage(path[0])
                    temp_mask = sitk.ReadImage(path[1])
                    temp_image_array = sitk.GetArrayFromImage(temp_data)
                    temp_mask_array = sitk.GetArrayFromImage(temp_mask)
            
                    directory = os.path.join(export_path,'images_quick_check',pat,path[1][:-5].split('\\')[-1])
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
                except:
                    warn('Something wrong with %s'%pat)
                    
        else:
            raise TypeError('The toolbox should be initialized with data_type = "nrrd" ')
      
    def extract_features(self,params_file,loggenabled=False): 
        
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
                        pat_features.update({'Patient':pat,'ROI':path[1].split('\\')[-1][:-5]}) 
                        feat_dictionary[key_number] = pat_features   
                        key_number+=1
                        
                except KeyboardInterrupt:
                    raise
                except:
                    warn('region : %s skipped'%pat)
            
            output_features = DataFrame.from_dict(feat_dictionary).T
            
            return output_features
        else:
            raise TypeError('The toolbox should be initialized with data_type = "nrrd"')
        
    def convert_to_nrrd(self,export_path,region_of_interest='all',image_type = np.int16):
        '''Convert DICOM dataset to the volume (nrrd) format
        export_path = ... export path where the converted nrrds will be placed
        region_of_interest = ... if you know exact name of the ROI you want to extract then write it with the ! character infront,
        eg. region_of_interest = !gtv1 , if you want to extract all the gtv's in the rtstructure eg. gtv1, gtv2, gtv_whatever then just specify 
        the stem word eg. region_of_interest = gtv, default value is region_of_interest ='all' , which mean that all ROI's in rtstruct will be extracted.
        
        '''
        if self._data_type == 'dcm':
            
            for pat,pat_path in tqdm(self,desc='Patients converted'):
                img_path = pat_path[0]
                rt_path = pat_path[1]
                try:
                    rt_structure,roi_list = self.__get_roi(region_of_interest,rt_path)
                except KeyboardInterrupt:
                    raise
                except:
                    roi_list=[]
                    warn('Error: ROI extraction failed for patient%s'%pat)
                    
                for roi in roi_list:
                    try:
                        image,mask = self.__get_binary_mask(img_path,rt_structure,roi,image_type) 
                        
                        export_dir = os.path.join(export_path,'converted_nrrds',pat)
                        
                        if not os.path.exists(export_dir):
                            os.makedirs(export_dir)
                        
                        image_file_name='image.nrrd'
                        mask_file_name='%s_mask.nrrd'%roi
                        sitk.WriteImage(image,os.path.join(export_dir,image_file_name)) # save image and binary mask locally
                        sitk.WriteImage(mask,os.path.join(export_dir,mask_file_name))
                    except KeyboardInterrupt:
                        raise    
                    except:
                        warn('Patients %s ROI : %s skipped'%(pat,roi))
                        
                        
        else:
            raise TypeError('Currently only conversion from dicom -> nrrd is available')
        


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
    
    def __read_scan(self,path):
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
            except:
                skiped_files.append(s)
                    
        try:
            scan.sort(key = lambda x: x.ImagePositionPatient[2])
        except:
            warn('Some problems with sorting scans')
                
        return scan,skiped_files
    
    def __get_pixel_values(self,scans,image_type): 
        try:
            # slopes = [s.get('RescaleSlope','n') for s in scans]
            
            # if set('n').intersection(set(slopes)):
            #     image=[]
            #     for s in scans:
            #         temp_intercept = 0#s[0x040,0x9096][0][0x040,0x9224].value
            #         temp_slope = 1#s[0x040,0x9096][0][0x040,0x9225].value
            #         image.append(s.pixel_array*temp_slope+temp_intercept)
            #     image = np.stack(image)
            #     return image.astype(np.int16)
            
            # else:    
            image = np.stack([s.pixel_array*s.RescaleSlope+s.RescaleIntercept for s in scans])
            return image.astype(image_type) 
        except:
            warn('Problems occured with rescaling intensities')
            image = np.stack([s.pixel_array for s in scans])
            return image.astype(image_type)

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
        except:
            return 'NaN'
###Some debugs
        
# parameters = {'data_path': r'**', 
#               'data_type': 'dcm',
#               'mask_names': [],
#               'image_only': False, 
#               'multi_rts_per_pat': True,
#               'image_names': ['image','volume','img','vol']}      
    
    
#Data1 = tool_box(**parameters) 
#Data1.convert_to_nrrd(r'**','all')  
#Data_MRI_nrrd = tool_box(data_path = r'**',data_type='nrrd')
#Data_MRI_nrrd.get_jpegs(r'**')
#parameters = r"**"
#features = Data_MRI_nrrd.extract_features(parameters,loggenabled=True)


#parameters = {'data_path': r'C:\Users\S.Primakov.000\Documents\GitHub\The_Dlab_toolbox\data\dcms', #path_to_your_data
#  'data_type': 'dcm'}  
#mri_dcms = tool_box(**parameters)
#dataset_description = mri_dcms.get_dataset_description('MRI')   