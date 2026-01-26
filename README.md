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

- Install python (Tested with Python 3.10)
- Download the package: Code (green button) > Download ZIP
- Unzip and and move directory to desire folder
- Using a command line Terminal, cd into the directory
- In the Terminal, type the following, enter after each line:

python3 -m venv perfmri_env
source perfmri_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Run
python PerfMRI.py


### Optional: segmentation support (ANTsPy)

PerfMRI can use ANTsPy for brain segmentation.
If you want segmentation features, try:

pip install antspyx

If installation fails, PerfMRI will still run but segmentation features will be disabled.

## To run PerfMRI
/path_to_program/PerfMRI.py
or 
python /path_to_program/PerfMRI.py


