# PerfMRI GUI
Program to analyze MRI DSC perfusion, BOLD breath-hold, and cerebrovascular reactivity datasets.

## Features

### Pre-processing
- Automask
- Slice-time correction
- Volume re-registration
- Signal detrending
- Signal scaling
- Spatial smoothing
- Temporal smoothing

### DSC Perfusion
- Relative perfusion metrics (AUC, TTP, FWHM, BAT)
- TTP, BAT maps have sub-TR temporal precision
- Automated AIF detection based on relative perfusion metrics
- Quantitative perfusion using SVD, oSVD, or model-based residue function (exponential)
- Tmax calculation

### Breath-Hold and CVR
- Input stimulus based on ON/OFF timing from a user-provided stimulus file:
  - One-column stimulus file (one value per TR)
  - Two-column stimulus file (time, value)
- Fast linear regression of BOLD signal against the stimulus
- Stimulus can be shifted left or right, with BOLD maps auto-recalculated to visualize the effect of lag
- Lag and tau analysis

### ROI 
- Drawing, loading and saving ROI
- Averaging metrics within ROI

### General Features
- Input formats: NIfTI, DICOM, AFNI
- (De-)oblique anatomical data to match functional obliquity
- Fast voxel navigation for data browsing
- Multiple colorscales available
- Automatic or manual color scale limits
- All maps can be automatically or manually exported
- Quick segmentation into grey matter, white matter, and CSF, subdivided into left and right hemisphere

## Installation (Tested on Mac Sequoia)

### Install Python 3.10:
    brew update
    brew install python@3.10 python-tk@3.10
### Create a virtual environment
    python3.10 -m venv perfmri_env
### Activate the virtual environment (bash & zsh)
    source perfmri_env/bin/activate
### Activate the virtual environment (tcsh & csh)
    source perfmri_env/bin/activate.csh
### Verify Python version (should be 3.10.x)
    python --version
### Verify Tk is installed
    python -m tkinter
### Upgrade pip 
    python -m pip install --upgrade pip
### Install dependencies
    python -m pip install -r requirements.txt
### Optional
#### For separating Left & Right hem.
    pip install antspyx
## Run PerfMRI
    python  PerfMRI.py


