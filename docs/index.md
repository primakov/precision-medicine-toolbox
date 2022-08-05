# Welcome to *precision-medicine-toolbox* documentation!
[![DOI](https://zenodo.org/badge/259937957.svg)](https://zenodo.org/badge/latestdoi/259937957)
[![License](https://img.shields.io/github/license/primakov/precision-medicine-toolbox)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/precision-medicine-toolbox/badge/?version=latest)](https://precision-medicine-toolbox.readthedocs.io/en/latest/?badge=latest)
[![CodeFactor](https://www.codefactor.io/repository/github/primakov/precision-medicine-toolbox/badge)](https://www.codefactor.io/repository/github/primakov/precision-medicine-toolbox)
[![PyPI](https://img.shields.io/pypi/v/precision-medicine-toolbox)](https://pypi.org/project/precision-medicine-toolbox/)  

*precision-medicine-toolbox* is an open-source python package for medical imaging data preparation for data science tasks. 
This package is aimed to provide a tool to curate the imaging data and to perform exploratory feature analysis.  

If you are using this toolbox, please, cite the [original paper](https://arxiv.org/abs/2202.13965):  
*Primakov, Sergey, Elizaveta Lavrova, Zohaib Salahuddin, Henry C. Woodruff, and Philippe Lambin. "Precision-medicine-toolbox: An open-source python package for facilitation of quantitative medical imaging and radiomics analysis." arXiv preprint arXiv:2202.13965 (2022).*

Currently, the toolbox has the following functionality:  
  
* **Dataset exploration.** This function gets the specified metadata from the DICOM files of the dataset and allows for exploration of the diversity degree of the imaging parameters..
* **Dataset quality check.** This function checks every scan in the dataset to be in line with the pre-defined requirements:  
    * imaging modality is correct,  
    * slice thickness is in the acceptable range,  
    * number of slices is in the acceptable range,  
    * all the slices have a target plane resolution,  
    * in-plane pixel spacing is in the acceptable range,  
    * reconstruction kernel for CT data is presented and is acceptable.  
* **Conversion of DICOM to NRRD.** This function allows for the conversion of DICOM (CT or MR) dataset into volume (NRRD format) dataset. 2D data is temporarily not supported.  
* **Basic image pre-processing.** This function performs basic image pre-processing steps, selected by the user; the following methods are available:  
    * N4 bias field correction,  
    * intensity rescaling, based on fat values or percentile values,  
    * histogram matching,  
    * intensities resampling,  
    * histogram equalization,  
    * Z-scoring, based on defined normalization coefficients or image-based values,  
    * image reshaping.  
* **Unrolling NRRD images & ROI masks into jpeg slices.** This function could be used for a quick check of the converted images or any existing NRRD/MHA dataset. It will generate the JPEG images for each ROI slice.  
* **Extracting of radiomics features.** Feature extraction procedure using pyradiomics to obtain the radiomics features for NRRD/MHA dataset.  
* **Basic analysis of radiomics features.** Export to Excel file of features basic statistics and statistical tests values and visualization (in .html report) of:  
    * features values distributions in binary classes,
    * Shapiro-Wilk test for normality check,
    * features mutual correlation (Spearman) matrix,
    * p-values (corrected) for Mann-Whitney test for features mean values in groups,
    * univariate ROC-curves for each feature,
    * volumetric analysis: volume-based precision-recall curve + features correlation with volume.
* **Binary classification metrics reporting.** Given true labels and predicted probabilities, this function performs:
    * classification metrics reporting,
    * confusion matrices and ROC curves plotting.

**Warning!** Not intended for clinical use!

## Code and documentation
*precision-medicine-toolbox* is an open-source package, the source code is available [online](https://github.com/primakov/precision-medicine-toolbox). 
The online documentation is available [here](http://precision-medicine-toolbox.readthedocs.io/). 
The functionality of the toolbox is illustrated in the tutorial [notebooks](https://github.com/primakov/precision-medicine-toolbox/tree/master/examples).
## 3rd-party packages used in *precision-medicine-toolbox*
Our package is using the existing quality tools for the key steps:

* pydicom (DICOM I/O),
* SimpleITK (image I/O and pre-processing),
* pyradiomics (features extraction).

See [requirements.txt](https://github.com/primakov/precision-medicine-toolbox/blob/master/requirements.txt) for more.
## Installation
Before use, install the dependencies from the requirements file:  
```
pip install -r requirements.txt   
```  
Then clone repository with the git client of your preference.

The latest version is also available at PyPi:
```
pip install precision-medicine-toolbox   
``` 
## Quick start
The following example illustrates how to initialize an object of a dataset class:  
```python
import os, sys
sys.path.append('path to precision-medicine-toolbox directory')
from pmtool.ToolBox import ToolBox

# set up parameters for your imaging dataset
parameters = {'data_path': 'root directory of the imaging data',
              'data_type': 'dcm', # DICOM data
              'multi_rts_per_pat': False # looks at 1 RTStruct/patient only
              }
my_dataset = ToolBox(**parameters)
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

Also you can see the list of the [contributors](https://github.com/primakov/precision-medicine-toolbox/graphs/contributors).
## License
This project is licensed under the BSD-3-Clause License 
(see the [LICENSE](https://github.com/primakov/precision-medicine-toolbox/blob/master/LICENSE) for the details).
## Acknowledgements  
The Precision Medicine department colleagues for their support and feedback.  
PyRadiomics for a reliable open-source tool for features extraction.  
Hugo Aerts et al. for the Lung1 dataset we used to demonstrate our functionality 
and TCIA for the publically available data.
