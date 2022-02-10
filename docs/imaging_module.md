# Imaging module tutorial

This tutorial shows how to convert DICOM into NRRD, check ROI segmentation, extract the radiomic features, and explore the imaging parameters.

Importing modules:


```python
import os,sys
sys.path.append(os.path.abspath(".."))
from tool_box import tool_box
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

%matplotlib inline
```

## Data download

The Lung1 dataset (Aerts et al., 2014) contains pretreatment scans of 422 non-small cell lung cancer (NSCLC) patients, as well as manually deliniated gross tumor volume (GTV) for each patient, and clinical outcomes. More information you can find on the dataset web page and in the corresponding paper https://doi.org/10.1038/ncomms5006. For consistency, we recommend you to download the data to the '../data/dcms' folder of this project. You can find the data and its description (and the original paper) following the link below:
https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics

To download the dataset, you might need to install the download manager: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-DownloadingtheNBIADataRetriever

Finally, you will have the '../data/dcms' folder with CT images (it will require 33 GB of disc space!).

## Dataset exploration

Setting the parameters:


```python
parameters = {'data_path': r'../data/dcms/', # path_to_your_DICOM_data
              'data_type': 'dcm', # original data format: DICOM
              'multi_rts_per_pat': False}   # when False, it will look only for 1 rtstruct in the patient folder, 
                                            # this will speed up the process, 
                                            # if you have more then 1 rtstruct per patient, set it to True
```

Initialize the dataset:


```python
data_dcms = tool_box(**parameters)
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.28it/s]
    

Get the default metainformation from the DICOM files and print first 10:


```python
dataset_description = data_dcms.get_dataset_description() 
dataset_description.head(10)
```

    Patients processed: 100%|████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.38s/it]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Modality</th>
      <th>SliceThickness</th>
      <th>PixelSpacing</th>
      <th>SeriesDate</th>
      <th>Manufacturer</th>
      <th>patient</th>
      <th>slice#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CT</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>20180209</td>
      <td>SIEMENS</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



You can check for the needed info by printing unique values of the tags:


```python
print('Unique modalities found: ', np.unique(dataset_description.Modality.values)[0])
```

    Unique modalities found:  CT
    

There are two pre-set parameter dictionaries for 'CT' and 'MRI' with the extended amount of parameters, parsed from the DICOM tags. These parameters are specific for these imaging modalities. You can also send a custom list of parameters. In this case the DICOM tags should have the same names as corresponding pydicom (DICOM keyword, https://pydicom.github.io/pydicom/dev/old/base_element.html).

Get the CT-specific metainformation from the DICOM files and print first 10:


```python
ct_dcms = tool_box(**parameters)
dataset_description = ct_dcms.get_dataset_description('CT') 
dataset_description.head(10)
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.33it/s]
    Patients processed: 100%|████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.10it/s]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientName</th>
      <th>ConvolutionKernel</th>
      <th>SliceThickness</th>
      <th>PixelSpacing</th>
      <th>KVP</th>
      <th>Exposure</th>
      <th>XRayTubeCurrent</th>
      <th>SeriesDate</th>
      <th>patient</th>
      <th>slice#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LUNG1-001</td>
      <td>B19f</td>
      <td>3.0</td>
      <td>[0.9765625, 0.9765625]</td>
      <td>120.0</td>
      <td>400</td>
      <td>80</td>
      <td>20180209</td>
      <td>LUNG1-001_20180209_CT_2</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot some of the scans parameters:


```python
sb.set(context='poster', style='whitegrid')

study_date = sorted([ 'Nan' if x=='' or x=='NaN' else str(x[0:4]) for x in list(dataset_description['SeriesDate'])])[2:]
conv_kernel =['Nan' if x=='' or x=='NaN' else x for x in list(dataset_description['ConvolutionKernel'])]
tube_current =[-1 if x=='' or x=='NaN' else x for x in list(dataset_description['XRayTubeCurrent'])]
exposure =[-1 if x=='' or x=='NaN' else x for x in list(dataset_description['Exposure'])]
ps = sorted([(x[0]) for x in list(filter(lambda x: x != 'NaN', dataset_description['PixelSpacing'].values))])
sl_th = sorted([str(x)[0:3] for x in list(filter(lambda x: x != 'NaN', dataset_description['SliceThickness'].values))])
figures,descriptions = [study_date,conv_kernel,tube_current,exposure,ps,sl_th],['Study Date','Conv Kernel','Tube Current','Exposure','Pixel spacing','Slice Thickness']

fig,ax = plt.subplots(2,3,figsize=(25,15))
for i in range(2):
    for j in range(3):
        ax[i,j].hist(figures.pop(0),alpha=0.7)
        ax[i,j].set_title(descriptions.pop(0),fontsize=20)
```


    
![png](output_19_0.png)
    


##  DICOM to NRRD conversion

To convert DICOM dataset to the volume (nrrd) format, set up the parameters:  

* '*export_path = ...*' export path where the converted NRRDs will be placed,  
* '*region_of_interest = ...*' if you know exact name of the ROI you want to extract, then write it with the '!' character in front, eg. '*region_of_interest = !gtv1*',
* if you want to extract all the GTVs in the rtstruct eg. '*gtv1*', '*gtv2*', '*gtv_whatever*', then just specify the stem word, eg. '*region_of_interest = gtv*', 
* default value is '*region_of_interest ='all'*', where all ROI's in rtstruct will be extracted.


```python
export_path =r'../data/' # the function will create 'converted_nrrd' folder in the specified directory       
```

Initialize the dataset (originally downloaded directory with DICOM files):


```python
data_dcms = tool_box(**parameters) 
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.35it/s]
    

Run the conversion:


```python
data_dcms.convert_to_nrrd(export_path, 'gtv')
```

    Patients converted: 100%|████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.23s/it]
    

## Quick check of the ROI's in the NRRD dataset

If you want to check your converted images & ROIs, or just any volumetric dataset (NRRD/MHA), you can use get_jpegs function of the toolbox. It will generate JPEG images for each slice with the image and overlap of the contour.

Initialize the dataset (converted NRRD files):


```python
data_ct_nrrd = tool_box(data_path = r'../data/converted_nrrds/', data_type='nrrd')
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3018.21it/s]
    

Run the visualization:


```python
data_ct_nrrd.get_jpegs(r'../data/') # the function will create 'images_quick_check' folder in the specified directory 
```

    Patients processed: 100%|████████████████████████████████████████████████████████████████| 5/5 [00:52<00:00, 10.56s/it]
    

Let's check one of the patients:


```python
from ipywidgets import interact
import numpy as np
from PIL import Image

def browse_images(images,names):
    n = len(images)
    def view_image(i):
        plt.figure(figsize=(20,10))
        plt.imshow(images[i])#, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Slice: %s' % names[i])
        plt.axis('off')
        plt.show()
    interact(view_image, i=(0,n-1))
    
for pat,_ in data_ct_nrrd:
    _,file_struct = [*os.walk(os.path.join('../data/images_quick_check/',pat))]
    root,images = file_struct[0],file_struct[2]
    imgs =[np.array(Image.open(os.path.join(root,img))) for img in images]
    print(pat)
    browse_images(imgs,images)
    break
```

    LUNG1-001_20180209_CT_2_GTV-1_mask
    


    interactive(children=(IntSlider(value=10, description='i', max=20), Output()), _dom_classes=('widget-interact'…


## Image pre-processing

Pre-processing with 'naive' parameters (consider they might not work for CT data):


```python
reference_image_path = '../data/converted_nrrds/LUNG1-001_20180209_CT_2/image.nrrd'
save_preprocessed_path = '../data/nrrd_preprocessed'

data_ct_nrrd.pre_process(ref_img_path = reference_image_path,
                         save_path = '../data/nrrd_preprocessed',
                         hist_match = True,         # boolean
                         subcateneus_fat = False,   # boolean
                         fat_value = 774,           # this is a dummy value, 
                                                    # you would need to find that value for each image
                         percentile_scaling = True, # boolean
                         binning = 255,             # this is a dummy value, it takes False or int 
                                                    # (# of bins for intensity resampling)
                         verbosity = True,          # boolean
                         z_score = False,           # boolean 
                         hist_equalize = False,     # boolean
                         norm_coeff = (1000.,500.), # these are dummy values, you would need to estimate real mu and sigma
                                                    # it takes None or tuple: (mu,sigma)
                         visualize = True)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_37_2.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_37_4.png)
    


    Histogram matching was applied
    


    
![png](output_37_6.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  186 resampled img:  50
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_37_8.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:29<01:59, 29.99s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_37_11.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_37_13.png)
    


    Histogram matching was applied
    


    
![png](output_37_15.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_37_17.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [00:58<01:26, 28.86s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_37_20.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_37_22.png)
    


    Histogram matching was applied
    


    
![png](output_37_24.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_37_26.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [01:26<00:57, 28.50s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_37_29.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_37_31.png)
    


    Histogram matching was applied
    


    
![png](output_37_33.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_37_35.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [01:54<00:28, 28.60s/it]
    

Changing the pre-processing parameters:


```python
#EXAMPLE 1: example of param file for zscore based on the image with visualization and verbosity
ex1 = {'ref_img_path': reference_image_path,
      'save_path': save_preprocessed_path + '_ex1',
      'verbosity': True,              #boolean
      'z_score': True,                #boolean
      'visualize': True}     
#EXAMPLE 2: example of param file for zscore based on the whole image with visualization and verbosity
ex2 = {'ref_img_path': reference_image_path,
        'save_path': save_preprocessed_path + '_ex2',
        'verbosity': True,              #boolean
        'z_score': True,                #boolean
        'norm_coeff': (1000.,500.),
        'visualize': True}     
#EXAMPLE 3: example of param file for histogramm matching with visualization and verbosity
ex3 = {'ref_img_path': reference_image_path,
        'save_path': save_preprocessed_path + '_ex3',
        'verbosity': True,              #boolean
        'hist_match': True,                 #boolean
        'visualize': True}  
#Example 4: example of param file for histogramm matching and image based z-score with visualization and verbosity
ex4 = {'ref_img_path': reference_image_path,
        'save_path': save_preprocessed_path + '_ex4',
        'verbosity': True,              #boolean
        'hist_match': True,                 #boolean
        'z_score': True,                #boolean
        'visualize': True} 
#EXAMPLE 5: EVERYTHING ENABLED (doesn't make sense, just an example)
ex5 = {'ref_img_path': reference_image_path,
      'save_path': save_preprocessed_path + '_ex5',
      'hist_match': True,              #boolean
      'subcateneus_fat': False,        #boolean
      'fat_value': 774,                #this is a dummy value, you would need to find that value for each image (i guess we would need masks, or if you will show me how it looks i might automate it (not sure))
      'percentile_scaling': True,      #boolean
      'binning': 255,                  #this is a dummy value, it takes False or int (# of bins for intensity resampling
      'verbosity': True,               #boolean
      'z_score': False,                #boolean 
      'hist_equalize':False,           #boolean
      #'norm_coeff': (1000.,500.),     #these are dummy values, you would need to estimate real mu and sigma, it takes None or tuple: (mu,sigma)
      'visualize': True}         
```

Image pre-processing with the parameters set:


```python
data_ct_nrrd.pre_process(**ex1)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3033
    Intensity values range:[-1024:3034]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_41_2.png)
    


    ROI MU = -741.3879128925836, sigma = 426.958591508985
    Z-score normalization applied based on image intensities, Mu=-741.3879128925836, sigma=426.958591508985
    


    
![png](output_41_4.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:03<00:14,  3.70s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_41_7.png)
    


    ROI MU = -755.2762267310341, sigma = 431.23438322078636
    Z-score normalization applied based on image intensities, Mu=-755.2762267310341, sigma=431.23438322078636
    


    
![png](output_41_9.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [00:07<00:10,  3.57s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_41_12.png)
    


    ROI MU = -676.2653498070263, sigma = 472.3594222375149
    Z-score normalization applied based on image intensities, Mu=-676.2653498070263, sigma=472.3594222375149
    


    
![png](output_41_14.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [00:10<00:07,  3.51s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_41_17.png)
    


    ROI MU = -676.2653498070263, sigma = 472.3594222375149
    Z-score normalization applied based on image intensities, Mu=-676.2653498070263, sigma=472.3594222375149
    


    
![png](output_41_19.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [00:13<00:03,  3.45s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_41_22.png)
    


    ROI MU = -676.2653498070263, sigma = 472.3594222375149
    Z-score normalization applied based on image intensities, Mu=-676.2653498070263, sigma=472.3594222375149
    


    
![png](output_41_24.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:17<00:00,  3.55s/it]
    


```python
data_ct_nrrd.pre_process(**ex2)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3033
    Intensity values range:[-1024:3034]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_42_2.png)
    


    Z-score normalization applied based on the whole image, Mu=1000.0, sigma=500.0
    


    
![png](output_42_4.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:03<00:13,  3.32s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_42_7.png)
    


    Z-score normalization applied based on the whole image, Mu=1000.0, sigma=500.0
    


    
![png](output_42_9.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [00:06<00:10,  3.38s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_42_12.png)
    


    Z-score normalization applied based on the whole image, Mu=1000.0, sigma=500.0
    


    
![png](output_42_14.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [00:09<00:06,  3.32s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_42_17.png)
    


    Z-score normalization applied based on the whole image, Mu=1000.0, sigma=500.0
    


    
![png](output_42_19.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [00:12<00:03,  3.12s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_42_22.png)
    


    Z-score normalization applied based on the whole image, Mu=1000.0, sigma=500.0
    


    
![png](output_42_24.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:16<00:00,  3.22s/it]
    


```python
data_ct_nrrd.pre_process(**ex3)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3033
    Intensity values range:[-1024:3034]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_43_2.png)
    


    Histogram matching was applied
    


    
![png](output_43_4.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:08<00:32,  8.07s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_43_7.png)
    


    Histogram matching was applied
    


    
![png](output_43_9.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [00:14<00:21,  7.23s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_43_12.png)
    


    Histogram matching was applied
    


    
![png](output_43_14.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [00:21<00:13,  6.85s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_43_17.png)
    


    Histogram matching was applied
    


    
![png](output_43_19.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [00:27<00:06,  6.70s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_43_22.png)
    


    Histogram matching was applied
    


    
![png](output_43_24.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:34<00:00,  6.81s/it]
    


```python
data_ct_nrrd.pre_process(**ex4)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3033
    Intensity values range:[-1024:3034]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_44_2.png)
    


    Histogram matching was applied
    


    
![png](output_44_4.png)
    


    ROI MU = -741.3879128925836, sigma = 426.958591508985
    Z-score normalization applied based on image intensities, Mu=-741.3879128925836, sigma=426.958591508985
    


    
![png](output_44_6.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:09<00:39,  9.77s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_44_9.png)
    


    Histogram matching was applied
    


    
![png](output_44_11.png)
    


    ROI MU = -739.89883285385, sigma = 426.0112759965821
    Z-score normalization applied based on image intensities, Mu=-739.89883285385, sigma=426.0112759965821
    


    
![png](output_44_13.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [00:18<00:27,  9.03s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_44_16.png)
    


    Histogram matching was applied
    


    
![png](output_44_18.png)
    


    ROI MU = -739.3388020524354, sigma = 426.30555036861983
    Z-score normalization applied based on image intensities, Mu=-739.3388020524354, sigma=426.30555036861983
    


    
![png](output_44_20.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [00:26<00:17,  8.53s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_44_23.png)
    


    Histogram matching was applied
    


    
![png](output_44_25.png)
    


    ROI MU = -739.3388020524354, sigma = 426.30555036861983
    Z-score normalization applied based on image intensities, Mu=-739.3388020524354, sigma=426.30555036861983
    


    
![png](output_44_27.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [00:34<00:08,  8.25s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_44_30.png)
    


    Histogram matching was applied
    


    
![png](output_44_32.png)
    


    ROI MU = -739.3388020524354, sigma = 426.30555036861983
    Z-score normalization applied based on image intensities, Mu=-739.3388020524354, sigma=426.30555036861983
    


    
![png](output_44_34.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:42<00:00,  8.45s/it]
    


```python
data_ct_nrrd.pre_process(**ex5)
```

      0%|                                                                                            | 0/5 [00:00<?, ?it/s]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3033
    Intensity values range:[-1024:3034]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_45_2.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_45_4.png)
    


    Histogram matching was applied
    


    
![png](output_45_6.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  177 resampled img:  51
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_45_8.png)
    


     20%|████████████████▊                                                                   | 1/5 [00:34<02:19, 34.80s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3019
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_45_11.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_45_13.png)
    


    Histogram matching was applied
    


    
![png](output_45_15.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  186 resampled img:  50
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_45_17.png)
    


     40%|█████████████████████████████████▌                                                  | 2/5 [01:05<01:36, 32.27s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_45_20.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_45_22.png)
    


    Histogram matching was applied
    


    
![png](output_45_24.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_45_26.png)
    


     60%|██████████████████████████████████████████████████▍                                 | 3/5 [01:35<01:02, 31.31s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_45_29.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_45_31.png)
    


    Histogram matching was applied
    


    
![png](output_45_33.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_45_35.png)
    


     80%|███████████████████████████████████████████████████████████████████▏                | 4/5 [02:04<00:30, 30.30s/it]

    ----------------------------------------
    Original image stats:
    Number of unique intensity values : 3829
    Intensity values range:[-1024:3071]
    Image intensity dtype: int16
    ----------------------------------------
    


    
![png](output_45_38.png)
    


    Intensity values rescaled based on the 95th percentile
    


    
![png](output_45_40.png)
    


    Histogram matching was applied
    


    
![png](output_45_42.png)
    


    Resampling with a step of:  7.882352941176471 Amount of unique values, original img:  213 resampled img:  27
    ----------------------------------------
    Intensity values resampled with a number of bins: 255
    ----------------------------------------
    


    
![png](output_45_44.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [02:32<00:00, 30.55s/it]
    

## PyRadiomics features extraction

In our toolbox, we are using PyRadiomics software (https://pyradiomics.readthedocs.io/en/latest/) to extract the features. You can read the full documentation for the currently stable version: https://pyradiomics.readthedocs.io/_/downloads/en/stable/pdf/.

We are using PyRadiomics parameters file customized for CT data:


```python
parameters = r"example_ct_parameters.yaml"
features = data_ct_nrrd.extract_features(parameters, loggenabled=True)
```

Printing the features for first 3 ROIs:


```python
features.head(3)
```

Writing the features into the Excel file (you will find it in the 'data/features' folder):


```python
writer = pd.ExcelWriter('../data/features/extracted_features.xlsx') 
features.to_excel(writer, 'Sheet1')
writer.save()
```


```python

```
