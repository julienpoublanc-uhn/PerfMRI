# PerfMRI GUI
Program to analyze MRI DSC perfusion, BOLD breath-hold, and cerebrovascular reactivity datasets.

## Features

### Pre-processing:
- Automask
- Slice-time correction
- Volume re-registration
- Signal detrending
- Signal scaling
- Spatial smoothing
- Temporal smoothing

### DSC Perfusion:
- Relative perfusion metrics (AUC, TTP, FWHM, BAT)
- TTP, BAT maps have sub-TR temporal precision
- Automated AIF detection based on relative perfusion metrics
- Quantitative perfusion using SVD, oSVD or model-based residue function (exponential)
- Tmax calculation

### Breath-Hold and CVR:
- Input stimulus based on ON/OFF timing from a user-provided stimulus file:
    - One-column stimulus file (one value per TR)
    - Two-column stimulus file (time, value)
- Fast linear regression of BOLD signal against the stimulus
- Stimulus can be shifted left or right, with BOLD maps auto-recalculated to visualize the effect of lag
- Lag and tau analysis

### General Features:
- Input formats: NIfTI, DICOM, AFNI
- (De-)oblique anatomical data to match functional obliquity
- Fast voxel navigation for data browsing
- Multiple colorscales available
- Automatic or manual color scale limits
- All maps can be automatically or manually exported
- Quick segmentation into grey matter, white matter and csf and subdivided into left and right hemisphere.

## Installation

- Tested with Python 3.10
- PerfMRI requires a few packages that can be installed as follow:

pip install numpy==1.24.2 scipy==1.15.3 matplotlib==3.10.3 nibabel==3.2.2 \
    nilearn==0.10.2 nipy==0.6.1 dicom2nifti==2.4.10 \
    pydicom==2.2.2 reorient-nii==1.0.0 antspyx==0.4.2

or using the requirements.txt file:
pip install -r requirements.txt

=
## To run PerfMRI
/path_to_program/PerfMRI.py
or 
python /path_to_program/PerfMRI.py


