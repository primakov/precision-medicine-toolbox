# Features module tutorial

This tutorial shows how to perform an exploratory analysis of the features data: to visualize features distribution in classes, plot the feature correlation matrix, check Mann-Whitney U-test p-values, plot univariate ROC (and calculate AUC) for each feature, perform volumetric analysis, and save all the scores. We will use PyRadiomics features as variables and binary survival label as an outcome.

Importing modules:


```python
import os,sys
sys.path.append(os.path.abspath(".."))
from features_set import features_set
import pandas as pd
```

## Required data

To run the tutorial, you will need the following files, available in the 'data/features' folder of the *pm_toolbox* repository:  
* '*extended_clinical_df.xlsx*' - this is the clinical data file provided with the Lung1 dataset (Aerts et al., 2014); we added binary variables of 1-, 1.5-, and 2-years survival, based on the presented data,
* '*extracted_features_full.xlsx*' - this is a file with the PyRadiomic features values (can be extracted in the imaging module tutorial) for **all** the patients from Lung1 dataset.

## Binary classes functionality

If you have 2 classes in your dataset, the following functionality is availabe for your analysis. We will divide Lung1 dataset by 1 year survival label, therefore having 2 classes.

Set up the parameters to get the data:


```python
parameters = {
    'feature_path': "../data/features/features.xlsx", # path to csv/xls file with features
    'outcome_path': "../data/features/extended_clinical_df.xlsx", #path to csv/xls file with outcome
    'patient_column': 'Patient', # name of column with patient ID
    'patient_in_outcome_column': 'PatientID', # name of column with patient ID in clinical data file
    'outcome_column': '1yearsurvival' # name of outcome column
}
```

Initialise the feature set (you will see a short summary):


```python
fs = features_set(**parameters)
```

    Number of observations: 149
    Class labels: ['0' '1']
    Classes balance: [0.4228187919463087, 0.5771812080536913]
    

Print some attributes of the feature set - first 10 patient IDs and first 10 feature names:


```python
print ('Patient IDs: ', fs._patient_name[:10])
print ('\nFeature names: ', fs._feature_column[:10])
```

    Patient IDs:  ['LUNG1-001_20180209_CT_2_GTV-1_mask', 'LUNG1-002_000000_GTV-1_mask', 'LUNG1-002_20180526_CT_1_GTV-1_mask', 'LUNG1-003_000000_GTV-1_mask', 'LUNG1-003_20180209_CT_1_GTV-1_mask', 'LUNG1-003_20180209_CT_1_GTV-2_mask', 'LUNG1-003_20180209_CT_1_GTV-3_mask', 'LUNG1-004_000000_GTV-1_mask', 'LUNG1-006_000000_GTV-1_mask', 'LUNG1-008_000000_GTV-1_mask']
    
    Feature names:  ['original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength']
    

Exclude patients with the unknown outcomes, if they are represented in the dataset (feature set parameters are re-initialised and short summary is printed again):


```python
fs.handle_nan(axis=0)
```

    Number of observations: 149
    Class labels: ['0' '1']
    Classes balance: [0.4228187919463087, 0.5771812080536913]
    

Print the head of the composed dataframe, containing both the variables and the outcome:


```python
fs._feature_outcome_dataframe.head(5)
```




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
      <th>original_shape_Elongation</th>
      <th>original_shape_Flatness</th>
      <th>original_shape_LeastAxisLength</th>
      <th>original_shape_MajorAxisLength</th>
      <th>original_shape_Maximum2DDiameterColumn</th>
      <th>original_shape_Maximum2DDiameterRow</th>
      <th>original_shape_Maximum2DDiameterSlice</th>
      <th>original_shape_Maximum3DDiameter</th>
      <th>original_shape_MeshVolume</th>
      <th>original_shape_MinorAxisLength</th>
      <th>...</th>
      <th>wavelet-LLL_gldm_HighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceLowGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LowGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceLowGrayLevelEmphasis</th>
      <th>ROI</th>
      <th>1yearsurvival</th>
    </tr>
    <tr>
      <th>Patient</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LUNG1-001_20180209_CT_2_GTV-1_mask</th>
      <td>0.732658</td>
      <td>0.548834</td>
      <td>46.151744</td>
      <td>84.090588</td>
      <td>95.336247</td>
      <td>83.186537</td>
      <td>95.425364</td>
      <td>96.104110</td>
      <td>155379.500000</td>
      <td>61.609664</td>
      <td>...</td>
      <td>14462.536758</td>
      <td>33.609142</td>
      <td>548084.071741</td>
      <td>0.002759</td>
      <td>0.000139</td>
      <td>0.267761</td>
      <td>2959.571494</td>
      <td>0.000058</td>
      <td>GTV-1_mask</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LUNG1-002_000000_GTV-1_mask</th>
      <td>0.878035</td>
      <td>0.755488</td>
      <td>70.110114</td>
      <td>92.801132</td>
      <td>116.931604</td>
      <td>101.833197</td>
      <td>104.316825</td>
      <td>125.674182</td>
      <td>358446.791667</td>
      <td>81.482668</td>
      <td>...</td>
      <td>13208.125493</td>
      <td>55.600107</td>
      <td>832176.248523</td>
      <td>0.004497</td>
      <td>0.000162</td>
      <td>0.188931</td>
      <td>1733.836805</td>
      <td>0.000053</td>
      <td>GTV-1_mask</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LUNG1-002_20180526_CT_1_GTV-1_mask</th>
      <td>0.878035</td>
      <td>0.755488</td>
      <td>70.110114</td>
      <td>92.801132</td>
      <td>116.931604</td>
      <td>101.833197</td>
      <td>104.316825</td>
      <td>125.674182</td>
      <td>358446.791667</td>
      <td>81.482668</td>
      <td>...</td>
      <td>13208.125493</td>
      <td>55.600107</td>
      <td>832176.248523</td>
      <td>0.004497</td>
      <td>0.000162</td>
      <td>0.188931</td>
      <td>1733.836805</td>
      <td>0.000053</td>
      <td>GTV-1_mask</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LUNG1-003_000000_GTV-1_mask</th>
      <td>0.544631</td>
      <td>0.356597</td>
      <td>25.559022</td>
      <td>71.674815</td>
      <td>56.639209</td>
      <td>83.528438</td>
      <td>62.265560</td>
      <td>84.011904</td>
      <td>34987.000000</td>
      <td>39.036358</td>
      <td>...</td>
      <td>9142.646956</td>
      <td>17.909008</td>
      <td>209143.444093</td>
      <td>0.002673</td>
      <td>0.000367</td>
      <td>0.402930</td>
      <td>2838.784544</td>
      <td>0.000191</td>
      <td>GTV-1_mask</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LUNG1-003_20180209_CT_1_GTV-1_mask</th>
      <td>0.544631</td>
      <td>0.356597</td>
      <td>25.559022</td>
      <td>71.674815</td>
      <td>56.639209</td>
      <td>83.528438</td>
      <td>62.265560</td>
      <td>84.011904</td>
      <td>34987.000000</td>
      <td>39.036358</td>
      <td>...</td>
      <td>9142.646956</td>
      <td>17.909008</td>
      <td>209143.444093</td>
      <td>0.002673</td>
      <td>0.000367</td>
      <td>0.402930</td>
      <td>2838.784544</td>
      <td>0.000191</td>
      <td>GTV-1_mask</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1220 columns</p>
</div>



Visualize feature values distribution in classes for the first 12 features (will pop up in an interactive .html report):


```python
fs.plot_distribution(fs._feature_column[:12])
```

Visualize mutual feature correlation coefficient (Spearman's) matrix for the first 12 features (in .html report):


```python
fs.plot_correlation_matrix(fs._feature_column[:12])
```

Visualize Mann-Whitney (Bonferroni corrected) p-values for binary classes test (in .html report):


```python
fs.plot_MW_p(fs._feature_column[:12])
```

Calculate the basic statistics for each feature (save the values in *data/features/features_basic_stats.xlsx*): number of NaN, mean, std, min, max; if applicable: MW-p, univariate ROC AUC, volume correlation:


```python
fs.calculate_basic_stats(volume_feature='original_shape_VoxelVolume')
```

Check the excel table:


```python
print('Basic statistics for each feature')
pd.read_excel('../data/features/features_basic_stats.xlsx')
```

    Basic statistics for each feature
    




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
      <th>Unnamed: 0</th>
      <th>NaN</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Min</th>
      <th>Max</th>
      <th>p_MW_corrected</th>
      <th>univar_auc</th>
      <th>volume_corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>original_shape_Elongation</td>
      <td>0</td>
      <td>0.720328</td>
      <td>0.161721</td>
      <td>0.062127</td>
      <td>0.974104</td>
      <td>0.540459</td>
      <td>0.517996</td>
      <td>0.037515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>original_shape_Flatness</td>
      <td>0</td>
      <td>0.559677</td>
      <td>0.154895</td>
      <td>0.047315</td>
      <td>0.856767</td>
      <td>0.548663</td>
      <td>0.516611</td>
      <td>0.099032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>original_shape_LeastAxisLength</td>
      <td>0</td>
      <td>32.055804</td>
      <td>15.983620</td>
      <td>6.643777</td>
      <td>85.495660</td>
      <td>0.000614</td>
      <td>0.686877</td>
      <td>0.973238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>original_shape_MajorAxisLength</td>
      <td>0</td>
      <td>61.595666</td>
      <td>35.336125</td>
      <td>13.611433</td>
      <td>240.822486</td>
      <td>0.000333</td>
      <td>0.705795</td>
      <td>0.842114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>original_shape_Maximum2DDiameterColumn</td>
      <td>0</td>
      <td>63.064956</td>
      <td>33.057474</td>
      <td>15.620499</td>
      <td>157.632484</td>
      <td>0.000773</td>
      <td>0.681525</td>
      <td>0.950909</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>wavelet-LLL_gldm_LargeDependenceLowGrayLevelEm...</td>
      <td>0</td>
      <td>1.646810</td>
      <td>8.820031</td>
      <td>0.001036</td>
      <td>79.946280</td>
      <td>0.323313</td>
      <td>0.524640</td>
      <td>0.070593</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>wavelet-LLL_gldm_LowGrayLevelEmphasis</td>
      <td>0</td>
      <td>0.007969</td>
      <td>0.039767</td>
      <td>0.000069</td>
      <td>0.369746</td>
      <td>0.001643</td>
      <td>0.515319</td>
      <td>-0.805811</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>wavelet-LLL_gldm_SmallDependenceEmphasis</td>
      <td>0</td>
      <td>0.313546</td>
      <td>0.177561</td>
      <td>0.009452</td>
      <td>0.755977</td>
      <td>0.003392</td>
      <td>0.654393</td>
      <td>-0.634677</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>wavelet-LLL_gldm_SmallDependenceHighGrayLevelE...</td>
      <td>0</td>
      <td>2431.019809</td>
      <td>1077.764252</td>
      <td>0.093515</td>
      <td>5302.913608</td>
      <td>0.022339</td>
      <td>0.619970</td>
      <td>-0.274001</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>wavelet-LLL_gldm_SmallDependenceLowGrayLevelEm...</td>
      <td>0</td>
      <td>0.000440</td>
      <td>0.000872</td>
      <td>0.000013</td>
      <td>0.007314</td>
      <td>0.000465</td>
      <td>0.654854</td>
      <td>-0.872534</td>
    </tr>
  </tbody>
</table>
<p>1218 rows × 9 columns</p>
</div>



Volume analysis will show you if your features have a high correlation to volume and if volume itself is a predictive feature in separation of 2 classes. You need to have a volume feature in your dataset and send it as a function parameter (in our case it is *'original_shape_VoxelVolume'*). 


```python
fs.volume_analysis(volume_feature='original_shape_VoxelVolume')
```

## Multi-class

If you have more than 2 classes in your dataset, the following functionality is availabe for your analysis. We will use a disease stage as a class label.

Set up the parameters to get the data:


```python
parameters = {
    'feature_path': "../data/features/extracted_features_full.xlsx", # path to csv/xls file with features
    'outcome_path': "../data/features/extended_clinical_df.xlsx", # path to csv/xls file with outcome
    'patient_column': 'Patient', # name of column with patient ID
    'patient_in_outcome_column': 'PatientID', # name of column with patient ID in clinical data file
    'outcome_column': 'Overall.Stage' # name of outcome column
}
```

Initialise the feature set:


```python
fs = features_set(**parameters)
```

    Number of observations: 149
    Class labels: ['I' 'II' 'IIIa' 'IIIb' 'nan']
    Classes balance: [0.24161073825503357, 0.09395973154362416, 0.2348993288590604, 0.4228187919463087, 0.006711409395973154]
    

Check the patients with the absent outcomes (because there is a NaN value among class labels in the data summary):


```python
fs._feature_outcome_dataframe[fs._feature_outcome_dataframe['Overall.Stage'].isnull()]
```




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
      <th>original_shape_Elongation</th>
      <th>original_shape_Flatness</th>
      <th>original_shape_LeastAxisLength</th>
      <th>original_shape_MajorAxisLength</th>
      <th>original_shape_Maximum2DDiameterColumn</th>
      <th>original_shape_Maximum2DDiameterRow</th>
      <th>original_shape_Maximum2DDiameterSlice</th>
      <th>original_shape_Maximum3DDiameter</th>
      <th>original_shape_MeshVolume</th>
      <th>original_shape_MinorAxisLength</th>
      <th>...</th>
      <th>wavelet-LLL_gldm_HighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LargeDependenceLowGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_LowGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis</th>
      <th>wavelet-LLL_gldm_SmallDependenceLowGrayLevelEmphasis</th>
      <th>ROI</th>
      <th>Overall.Stage</th>
    </tr>
    <tr>
      <th>Patient</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LUNG1-272_000000_GTV-1_mask</th>
      <td>0.713259</td>
      <td>0.487347</td>
      <td>14.2573</td>
      <td>29.254915</td>
      <td>25.495098</td>
      <td>31.400637</td>
      <td>35.608988</td>
      <td>37.868192</td>
      <td>5713.416667</td>
      <td>20.86634</td>
      <td>...</td>
      <td>11955.686684</td>
      <td>27.442472</td>
      <td>349332.116971</td>
      <td>0.002377</td>
      <td>0.00028</td>
      <td>0.233726</td>
      <td>2319.671592</td>
      <td>0.000215</td>
      <td>GTV-1_mask</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 1220 columns</p>
</div>



Exclude the patients with the unknown outcomes:


```python
fs.handle_nan(axis=0)
```

    Number of observations: 148
    Class labels: ['I' 'II' 'IIIa' 'IIIb']
    Classes balance: [0.24324324324324326, 0.0945945945945946, 0.23648648648648649, 0.42567567567567566]
    

Visualize feature values distribution in classes for the first 12 features (in .html report):


```python
fs.plot_distribution(fs._feature_column[:12])
```

Visualize feature values distribution in classes for the first 12 features only for the selected classes (in .html report):


```python
fs.plot_distribution(fs._feature_column[:12], ['I', 'IIIb'])
```

Visualize mutual feature correlation coefficient (Spearman's) matrix for the first 12 features (in .html report):


```python
fs.plot_correlation_matrix(fs._feature_column[:12])
```

Even though we have more than 2 classes and cannot perform Mann-Whitney test for all the classes, we can perform it for the pairs of the classes (result in .html report):


```python
fs.plot_MW_p(fs._feature_column[:100], ['I', 'IIIb'])
```

The same situation is with univariate analysis - we can perform it for the selected pairs of the classes:


```python
fs.plot_univariate_roc(fs._feature_column[:100], ['I', 'IIIb'])
```

Calculate the basic statistics for each feature (save the values in *data/features/features_basic_stats.xlsx*); for multi-class analysis the following are available: number of NaN, mean, std, min, max


```python
fs.calculate_basic_stats(volume_feature='original_shape_VoxelVolume')
```

Check the table:


```python
print('Basic statistics for each feature')
pd.read_excel('../data/features/features_basic_stats.xlsx')
```

    Basic statistics for each feature
    




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
      <th>Unnamed: 0</th>
      <th>NaN</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Min</th>
      <th>Max</th>
      <th>p_MW_corrected</th>
      <th>univar_auc</th>
      <th>volume_corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>original_shape_Elongation</td>
      <td>0</td>
      <td>0.720328</td>
      <td>0.161721</td>
      <td>0.062127</td>
      <td>0.974104</td>
      <td>0.540459</td>
      <td>0.517996</td>
      <td>0.037515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>original_shape_Flatness</td>
      <td>0</td>
      <td>0.559677</td>
      <td>0.154895</td>
      <td>0.047315</td>
      <td>0.856767</td>
      <td>0.548663</td>
      <td>0.516611</td>
      <td>0.099032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>original_shape_LeastAxisLength</td>
      <td>0</td>
      <td>32.055804</td>
      <td>15.983620</td>
      <td>6.643777</td>
      <td>85.495660</td>
      <td>0.000614</td>
      <td>0.686877</td>
      <td>0.973238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>original_shape_MajorAxisLength</td>
      <td>0</td>
      <td>61.595666</td>
      <td>35.336125</td>
      <td>13.611433</td>
      <td>240.822486</td>
      <td>0.000333</td>
      <td>0.705795</td>
      <td>0.842114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>original_shape_Maximum2DDiameterColumn</td>
      <td>0</td>
      <td>63.064956</td>
      <td>33.057474</td>
      <td>15.620499</td>
      <td>157.632484</td>
      <td>0.000773</td>
      <td>0.681525</td>
      <td>0.950909</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>wavelet-LLL_gldm_LargeDependenceLowGrayLevelEm...</td>
      <td>0</td>
      <td>1.646810</td>
      <td>8.820031</td>
      <td>0.001036</td>
      <td>79.946280</td>
      <td>0.323313</td>
      <td>0.524640</td>
      <td>0.070593</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>wavelet-LLL_gldm_LowGrayLevelEmphasis</td>
      <td>0</td>
      <td>0.007969</td>
      <td>0.039767</td>
      <td>0.000069</td>
      <td>0.369746</td>
      <td>0.001643</td>
      <td>0.515319</td>
      <td>-0.805811</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>wavelet-LLL_gldm_SmallDependenceEmphasis</td>
      <td>0</td>
      <td>0.313546</td>
      <td>0.177561</td>
      <td>0.009452</td>
      <td>0.755977</td>
      <td>0.003392</td>
      <td>0.654393</td>
      <td>-0.634677</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>wavelet-LLL_gldm_SmallDependenceHighGrayLevelE...</td>
      <td>0</td>
      <td>2431.019809</td>
      <td>1077.764252</td>
      <td>0.093515</td>
      <td>5302.913608</td>
      <td>0.022339</td>
      <td>0.619970</td>
      <td>-0.274001</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>wavelet-LLL_gldm_SmallDependenceLowGrayLevelEm...</td>
      <td>0</td>
      <td>0.000440</td>
      <td>0.000872</td>
      <td>0.000013</td>
      <td>0.007314</td>
      <td>0.000465</td>
      <td>0.654854</td>
      <td>-0.872534</td>
    </tr>
  </tbody>
</table>
<p>1218 rows × 9 columns</p>
</div>



Volume analysis (result in .html report):


```python
fs.volume_analysis(volume_feature='original_shape_VoxelVolume')
```
