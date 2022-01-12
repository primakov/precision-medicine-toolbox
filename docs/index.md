# Welcome to *pm_toolbox* documentation!
[![License](https://img.shields.io/github/license/precision-medicine-um/precision_medicine_toolbox)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/radiomics-training/badge/?version=latest)](https://radiomics-training.readthedocs.io/en/latest/?badge=latest)

The *pm_toolbox* (precision_medicine_toolbox) is an open-source python package for medical imaging data 
preparation for data science tasks. 
This package is aimed to provide a tool to curate the imaging data 
and to perform exploratory feature analysis.  

Currently, the toolbox has the following functionality:  
  
* **Conversion of DICOM to nrrd.** This function allows for the conversion of DICOM (CT or MR) dataset into volume (nrrd format) dataset. 2D data is temporarily not supported.  
* **Unrolling nrrd images & roi masks into jpeg slices.** This function could be used for a quick check of the converted images or any existing nrrd/mha dataset. It will generate the jpeg images for each ROI slice.  
* **Extracting of radiomics features.** Feature extraction procedure using pyradiomics to obtain the radiomics features for nrrd/mha dataset.  
* **Basic analysis of radiomics features.** Export to excel file of features basic statistics and statistical tests values and visualization (in .html report) of:  
    * features values distributions in binary classes,
    * features mutual correlation (Spearman) matrix,
    * p-values (Bonferroni corrected) for Mann-Whitney test for features mean values in groups,
    * univariate ROC-curves for each feature,
    * volumetric analysis: volume-based precision-recall curve + features correlation with volume.

**Warning!** Not intended for clinical use!

## Code and documentation
The *pm_toolbox* is an open-source package, the source code is available [online](https://github.com/precision-medicine-um/precision_medicine_toolbox). 
The online documentation is available [here](http://precision_medicine_toolbox.readthedocs.io/). 
The functionality of the toolbox is illustrated in the tutorial [notebooks]().
## 3rd-party packages used in the PM_toolbox
Our package is using the existing quality tools for the key steps:

* pydicom (DICOM I/O),
* SimpleITK (image I/O and pre-processing),
* pyradiomics (features extraction).

See [requirements.txt](https://github.com/precision-medicine-um/precision_medicine_toolbox/blob/master/requirements.txt) for more.
## Installation
Before use, install the dependencies from the requirements file:  
```
pip install -r requirements.txt   
```  
Then clone repository with the git client of your preference.
## Quick start
The following example illustrates how to initialize an object of a dataset class:  
```python
import os, sys
sys.path.append('path to pm_toolbox directory')
from tool_box import tool_box

# set up parameters for your imaging dataset
parameters = {'data_path': 'root directory of the imaging data',
              'data_type': 'dcm', # DICOM data
              'multi_rts_per_pat': False # looks at 1 RTStruct/patient only
              }
my_dataset = tool_box(**parameters)
```
## Contributing
You can contribute to this package at our GitHub by:  

* reporting the issues,
* giving us feedback for the code and the documentation via suggestions/comments:
    * directly in the Pull request,
    * writing and leaving a comment in the Conversation tab,
    * sending an e-mail to authors.
## Authors and citation
Initial and main developers:  

* Sergey Primakov [@primakov](https://github.com/primakov)
* Lisa Lavrova [@lavrovaliz](https://github.com/lavrovaliz)

Also you can see the list of the [contributors](https://github.com/precision-medicine-um/precision_medicine_toolbox/graphs/contributors).
## License
This project is licensed under the BSD-3-Clause License 
(see the [LICENSE](https://github.com/precision-medicine-um/precision_medicine_toolbox/blob/master/LICENSE) for the details).
## Acknowledgements  
The Precision Medicine department colleagues for their support and feedback.  
PyRadiomics for a reliable open-source tool for features extraction.  
Hugo Aerts et al. for the Lung1 dataset we used to demonstrate our functionality 
and TCIA for the publically available data.
