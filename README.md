# The Dlab toolbox

![Python version](https://img.shields.io/badge/python-3.6-green.svg),![Python version](https://img.shields.io/badge/python-3.7-green.svg)

## Description
Currently the toolbox has following functionality:

* <b>Conversion of DICOM to nrrd</b>
This function allows for convertion of DICOM(CT or MR)dataset into volume (nrrd format) dataset. 2D data is temporarily not supported.

* <b>Unrolling nrrd images&roi masks into jpeg slices </b>
This function could be used for quick check of the converted images, or any exicting nrrd/mha dataset. It will generate the jpeg images for each ROI slice.

* <b>Extracting of radiomics features </b>
Feature extraction procedure using [pyradiomics](https://github.com/Radiomics/pyradiomics) to obtain the radiomics features for nrrd/mha dataset.

* <b>Basic analysis of radiomics features </b>
Export to excel file of features basic statistics and statistical tests values and visualization (in .html report) of:  
  * features values distributions in binary and multiple classes,  
  * features mutual correlation (Spearman's) matrix,  
  * p-values (Bonferroni corrected) for Mann-Whitney test for features mean values in classes (binary, for multiple classes you need to select 2 classes to compare),   
  * univariate ROC-curves for each feature (binary, for multiple classes you need to select 2 classes),  
  * volume analysis: volume-based precision-recall curve + features correlation with volume.


For more information check out the <b>examples.ipynb</b>.
Examples of pyradiomics parameters for feature extraction can be found [here](https://github.com/Radiomics/pyradiomics/tree/master/examples/exampleSettings)


## Installation

Before use install the dependencies from the requirements file:

```
pip install -r requirements.txt
```


