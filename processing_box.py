# -*- coding: utf-8 -*-
"""
Created on Sun Apr 5 22:40:52 2020
@author: S.Primakov
s.primakov@maastrichtuniversity.nl

"""
import os,re,cv2
import pydicom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from collections import OrderedDict
from tqdm import tqdm
import csv
from data_set import data_set


is_in_list = lambda names_list, name: any([True for v in names_list if re.search(v,name.lower())])

class processing_box(data_set):
      
    def __normalize_image_zscore(self,image,norm_coeff):
        img= image - norm_coeff[0]
        img= image/norm_coeff[1]
        return img
    
    def __normalize_image_zscore_per_image(self,image):   ##Zscore based on the masked region intensities
        image_s = image.copy()
        mu = np.mean(image_s.flatten())
        sigma = np.std(image_s.flatten())
        if self.verbosity:
            print('ROI MU = %s, sigma = %s'%(mu,sigma))
        img= (image - mu)/sigma
        return img,mu,sigma
    
    def __resample_intensities(self,orig_img,bin_nr):            ## Intensity resampling whole image/or masked region
        v_count=0
        img_list=[]
        filtered = orig_img.copy()
        if np.min(orig_img.flatten())<0:
            filtered+=np.min(orig_img.flatten())
        resampled = np.zeros_like(filtered)
        max_val_img = np.max(filtered.flatten())
        step = max_val_img/bin_nr

        for st in np.arange(step,max_val_img+step,step):
            resampled[(filtered<=st)&(filtered>=st-step)] = v_count
            v_count+=1
        if self.verbosity:
            print("Resampling with a step of: ",step ,'Amount of unique values, original img: ',len(np.unique(orig_img.flatten())),'resampled img: ',len(np.unique(resampled.flatten())))

        if len(np.unique(resampled.flatten())) <255:
            return np.array(resampled,dtype=np.uint8)
        else:
            return np.array(resampled,dtype=np.uint16)
    
    
    
    def __intensity_scaling(self,img,fat_int_value= None,method=''):
        if method == 'fat' and fat_int_value:
            filtered = img.copy()
            if np.min(img.flatten())<0:
                filtered+=np.min(img.flatten())
            filtered = (1.0*filtered)/(1.0*np.max(img.flatten()))
            filtered = 1.0*fat_int_value*filtered
            return  np.array(filtered,np.uint16)

        elif method =='95th':
            filtered = img.copy()
            if np.min(img.flatten())<0:
                filtered+=np.min(img.flatten())
            perc_95 = np.percentile(filtered.flatten(),95)    
            filtered = (1.0*filtered)/(1.0*np.max(img.flatten()))
            filtered = 1.0*perc_95*filtered
            return  np.array(filtered,np.uint16)
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
    
    
    def __preprocessing_function(self, img, mask, ref_img, z_score, norm_coeff, hist_match,hist_equalize):
        if self.verbosity:
            unique_number_of_intensity = np.unique(img.flatten())
            min_int,max_int = np.min(img.flatten()),np.max(img.flatten())
            img_type = img.dtype
            print('-'*40)
            print('Original image stats:')
            print('Number of unique intensity values :',len(unique_number_of_intensity))
            print('Intensity values range:[%s:%s]'%(min_int,max_int))
            print('Image intensity dtype:',img_type)
            print('-'*40)
        if self.visualize:      
            plt.figure(figsize=(12,12))
            plt.imshow(img[int(len(img)/2.),...], cmap='bone')
            plt.title('original image')
            plt.show()

        if self.subcateneus_fat:
            img = self.__intensity_scaling(np.squeeze(img), self.fat_value,method='fat')
            if self.verbosity:
                print('Intensity values rescaled based on the subcateneus fat') 
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('subcateneus fat scaling')
                plt.show()
                
        if self.percentile_scaling:
            img = self.__intensity_scaling(np.squeeze(img), method='95th')
            if self.verbosity:
                print('Intensity values rescaled based on the 95th percentile')
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('95th percentile scaling')
                plt.show() 
                
        if hist_match:
            img = self.__histogram_matching(np.squeeze(img),np.squeeze(ref_img))
            if self.verbosity:
                print('Histogram matching was applied')
            if self.visualize:
                plt.figure(figsize=(22,12))
                plt.subplot(121)
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('Histogram matched')
                plt.subplot(122)
                plt.imshow(ref_img[int(len(img)/2.),...], cmap='bone')
                plt.title('Ref image')
                plt.show()  
                
        if binning:
            img = self.__resample_intensities(img,binning)
            if self.verbosity:
                print('-'*40)
                print('Intensity values resampled with a number of bins: %s'%binning)
                print('-'*40)
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('%s bins resampled'%binning)
                plt.show()

        if hist_equalize and img.dtype == np.uint8:
            equalized = [cv2.equalizeHist(np.squeeze(x).astype(np.uint8)) for x in img] 
            img = np.stack(equalized)
            if self.verbosity:
                print('Histogram equalization applied')
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('Histogram equalize')
                plt.show()

        if z_score and norm_coeff:
            img = self.__normalize_image_zscore(img,norm_coeff)
            if self.verbosity:
                print('Z-score normalization applied based on the whole image, Mu=%s, sigma=%s'%(norm_coeff))
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('Z-score normalization applied')
                plt.show()

        elif z_score:
            img,mu,sigma = self.__normalize_image_zscore_per_image(img)
            if self.verbosity:
                print('Z-score normalization applied based on image intensities, Mu=%s, sigma=%s'%(mu,sigma))
            if self.visualize:
                plt.figure(figsize=(12,12))
                plt.imshow(img[int(len(img)/2.),...], cmap='bone')
                plt.title('Z-score normalization applied')
                plt.show()

        return img 
    
    def pre_process(self, params, ref_img_path, save_path,
                    z_score, norm_coeff, hist_match, hist_equalize,
                    binning, percentile_scaling, subcataneus_fat, fat_value, verbosity, visualize):
        ref_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(ref_img_path))
        for i,pat in tqdm(self): 
            image = sitk.ReadImage(pat[0])
            mask = sitk.ReadImage(pat[1])
            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)
            ref_image_arr = ref_img_arr.copy()
            #run the pre-processing
            pre_processed_arr = self.__preprocessing_function(image_array,mask_array,ref_img_arr,z_score, norm_coeff,hist_match,hist_equalize,binning)
            
            export_dir = os.path.join(save_path,i)
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            pre_processed_image = sitk.GetImageFromArray(pre_processed_arr)
            pre_processed_image.CopyInformation(image)
            sitk.WriteImage(pre_processed_image,os.path.join(export_dir,'pre_processed_img.nrrd')) # save image and binary mask locally
            sitk.WriteImage(mask,os.path.join(export_dir,'mask.nrrd'))

        with open(os.path.join(save_path,'pre-processing_parameters.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Pre-processing parameters:'])
            for key, value in params.items():
                writer.writerow([str(str(key)+': '+str(value))])
