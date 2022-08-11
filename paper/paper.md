---
title: '`Precision-medicine-toolbox`: An open-source package for facilitation of quantitative medical imaging and radiomics analysis'
tags:
  - Python
  - medical imaging
  - radiomics
  - DICOM
  - features
  - statistical analysis
  - pre-processing
authors:
  - name: Elizaveta Lavrova
    orcid: 0000-0003-2751-790X
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2"
  - name: Sergey Primakov
    orcid: 0000-0002-3856-9740
    equal-contrib: true
    affiliation: 1
  - name: Zohaib Salahuddin
    orcid: 0000-0002-9900-329X
    affiliation: 1
  - name: Manon Beuque
    orcid: 0000-0001-6804-5908
    affiliation: 1
  - name: Damon Verstappen
    affiliation: 1
  - name: Henry C Woodruff
    orcid: 0000-0001-7911-5123
    affiliation: "1, 3"
  - name: Philippe Lambin
    orcid: 0000-0001-7961-0191
    affiliation: "1, 3"
affiliations:
 - name: Department of Precision Medicine, GROW School for Oncology, Maastricht University, Maastricht, The Netherlands
   index: 1
 - name: GIGA Cyclotrone Research Center, University of Liege, Liege, Belgium
   index: 2
 - name: Department of Radiology and Nuclear Medicine, Maastricht University Medical Centre, Maastricht, The Netherlands
   index: 3
date: 11 August 2022
bibliography: paper.bib
---

# Summary

Medical image analysis plays a key role in precision medicine, allowing clinicians to non-invasively 
identify phenotypic abnormalities and trends in routine clinical assessment. With the exponential increase 
of available imaging data and the advances of machine learning, state-of-the-art data science 
and engineering techniques are being applied to quantify  medical images and treat them as rich data sources. 
Data curation and pre-processing of medical images are critical steps in the quantitative medical image 
analysis that can have a significant impact on the resulting performance of machine learning models.
We introduce the `Precision-medicine-toolbox`, allowing clinical and junior researchers to perform 
data curation, image pre-processing, handcrafted radiomics extraction using open source software, 
and feature exploration tasks with a customizable Python package. With this open-source tool we aim 
to facilitate the crucial data preparation and exploration steps, bridge the gap between the currently 
existing packages, and improve the reproducibility of quantitative medical imaging research.

# Statement of need

The concept of precision medicine is rising in popularity, which aims to enhance individual patient care 
by identifying subgroups of patients within a disease group using genotypic and phenotypic data, 
consequently targeting the disease with more efficient treatment [@Niu2019-gw]. 
Medical image analysis plays a key role in precision medicine as it allows the clinicians to non-invasively 
identify (radiological) phenotypes [@Acharya2018-zi]. 

The number of medical imaging data to analyze is rising rapidly and physicians struggle to cope with 
the increasing demand. Moreover, multiple studies have shown that there is significant inter-observer 
variability when performing various clinical tasks [@Kinkel2000-dd; @Luijnenburg2010-ls]. 
Hence, there is a need for medical image analysis tools that can aid clinicians in meeting the challenges 
of rising demand and better clinical performance, while reducing variability and costs. At the heart of 
these tools will be advanced quantitative imaging analysis, such as handcrafted radiomics and deep learning. 
Handcrafted radiomics is the high-throughput extraction of pre-defined high-dimensional quantitative 
image features and their correlation with biological and clinical outcomes using machine learning methods 
[@Lambin2012-qo]. Deep learning automatically learns representative image features from the high 
dimensional image data without the need of feature engineering by using non-linear modules that constitute 
a neural network [@Schmidhuber2015-la]. As illustrated by \autoref{fig:1}, the field of quantitative image analysis 
is expanding due to increases in computational power and availability of large quantities of multimodal data 
[@Oren2020-eb; @Aggarwal2021-cm]. Moreover, it has demonstrated promising results in 
various clinical applications [@Tagliafico2020-jp; @Zhang2017-so; @Wang2021-zx; @Mu2020-pw]. 
As with many nascent technologies, high-throughput quantitative image analysis suffers from a lack of 
standardization, e.g. in the image domain (different vendors, acquisition and reconstruction protocols, 
pre-processing), or different definitions of handcrafted features (such as shape, intensity, and texture 
features), which initiatives such as the image biomarker standardization initiative (IBSI) try to counter 
[@Zwanenburg2020-ca]. 
The spread of widely used open-source software such as `Pyradiomics`, allows the extraction of IBSI-compliant 
handcrafted radiomics features [@Van_Griethuysen2017-ph]. 

![Number of publications, by year, containing keywords ((‘radiomics’ OR ‘deep learning’) AND ‘medical imaging’) in PubMed database (https://pubmed.ncbi.nlm.nih.gov/?term=(%E2%80%98radiomics%E2%80%99%20OR%20%E2%80%98deep%20learning%E2%80%99)%20AND%20%E2%80%98medical%20imaging%E2%80%99&timeline=expanded).\label{fig:1}](figure_1.png)

Data curation and the pre-processing of medical images are time-consuming and critical steps in the 
radiomics workflow that can have a significant impact on the resulting model performance 
[@Hosseini2021-bm; @Zhang2019-ha]. These steps may be performed manually or using lower level 
Python libraries such as `Numpy` [@Van_der_Walt2011-mn], 
`Pandas` [@McKinney2011-fb], `Pydicom` [@Mason2011-kt], `Scikit-image` [@Van_der_Walt2014-ih], 
`Scikit-learn` [@Kramer2016-zr], `SimpleITK` [@Yaniv2018-nh], `Nibabel` [@Brett2020-bm], 
or `Scipy` [@Virtanen2020-jy]. As most current data curation workflows necessitate time-consuming human 
input, this step becomes an error-prone bottleneck and adds to the current reproducibility problem. 
Several published works have led the IBSI to also emphasize the need for image processing before 
the extraction of radiomics features. Moreover, it is important to perform an exploratory analysis 
to understand the link between the data used as input in a machine learning model with the outcome 
it has to predict. While there are  tools available for the implementation of the radiomics 
pipeline such as `Nipype` [@Gorgolewski2016-vf], `Pymia` [@Jungo2021-fg], 
and `MONAI` [@MONAI_Consortium2020-kr], there is also the need for a tool that allows for the systematic 
and standardized data curation, image pre-processing, and feature exploration during the development 
phase of the study.

We introduce the open-source `Precision-medicine-toolbox` that facilitates data curation, image pre-processing, 
and feature exploration using customizable Python scripts. This toolbox will also benefit researchers 
without a strong programming background, allowing them to implement these steps and 
increase the reproducibility of quantitative medical imaging research. The toolbox was utilized and tested 
during the development of multiple projects including automatic lung tumor segmentation 
on the CT [@Primakov2021-cs], repeatability of breast MRI radiomic features [@Granzier2021-zl], 
and radiomics-based diagnosis of multiple sclerosis [@Lavrova2021-tf]. 
Based on feedback from the community of users, improvements and more functionality will be added 
to the toolbox with time. 

# Acknowledgements

The authors would like to thank the Precision Medicine department colleagues and external users for 
the feedback, Mart Smidt for testing the tool on the different data, PyRadiomics for a reliable 
open-source tool for features extraction, Hugo Aerts et al. for the Lung1 dataset we used 
to demonstrate our functionality, and The Cancer Imaging Archive for the publically available data.

# References
