# PerfMRI GUI
Program to analyse MRI DSC perfusion, BOLD breath hold and cerebrovascular reactivity dataset.

## Features
Pre-processing:
- Automask
- Slice-time correction
- Volume re-registration
- Signal detrending
- Signal scaling
- Spatial smoothing
- Temporal smoothing

DSC perfusion:
- Relative perfusion metrics (auc,ttp,fwhm,bat)
- ttp, bat, bat maps are in between TR precision
- Automated AIF search based on the relative perfusion metrics
- Quantitative perfusion using circular deconvolution or model based residue function (exponential)
- Tmax calculation 

Breath Hold and CVR
- Input stimulus based on ON/OFF timing input user, a one column stimulus file (one value at each TR)
    or two columns stimulus file (time,value)
- Fast linear regression of BOLD signal by the stimulus
- Stimulus can be shifted left and right with BOLD map automated re-calculated to visualize effect of the lag.
- Lag and tau analysis

General features
- Input files: nifti, dicom, AFNI
- (De)-oblique the anatomical to match the fuctional obliquity
- Quick voxel navigation for data browsing
- Multiple colorscales available
- Automated color scale limits
- All maps automaticaly or manually exported 


## Installation
- Tested with Python 3.10
- Before running this program, some packages need to be installed:
numpy 
scipy 
matplotlib 
nibabel 
nilearn 
nipy 
dicom2nifti 
pydicom 
reorient-nii

Here is the command lines with the specific versions that works together for me:
pip install numpy==1.24.2 scipy==1.10.1 matplotlib==3.9.0 nibabel==3.2.2 nilearn==0.10.2 nipy==0.6.1 dicom2nifti==2.4.10 pydicom==1.4.2 reorient-nii==1.0.0

## To run PerfMRI
/path_to_program/PerfMRI.py
or 
python /path_to_program/PerfMRI.py


