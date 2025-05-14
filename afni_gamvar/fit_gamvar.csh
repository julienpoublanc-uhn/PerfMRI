#!/bin/tcsh

cd $1
set afni_gamvar = `dirname "$0"`


set tr = `3dinfo -TR conc.nii.gz`

# Find start of bolus, end of bolus and end of recirculation
3dTstat -prefix conc_mean.nii.gz -mean conc.nii.gz
$afni_gamvar/3dmaskave -q -mask conc_mean.nii.gz conc.nii.gz > conc_ave.1D

set pmax = `3dTstat -argmax -prefix stdout: conc_ave.1D\'`

# Bolus start time
set b = `3dTstat -basepercent 15 -onset -prefix stdout: conc_ave.1D\'`

# Bolus end time
set c = `3dTstat -basepercent 15 -offset -prefix stdout: conc_ave.1D\'`

# Fit a gamma through the concentration

# Start time of the gamma variate: 6 sec before $b
set t1 = `echo "${tr}*${b} - 6" | bc -l`
# Start time of the gamma variate: 12 sec after $b
set t2 = `echo "${tr}*${b} + 12" | bc -l`


$afni_gamvar/3dNLfim -input conc.nii.gz\[0..$c\] -ignore 0 -mask conc.nii.gz\[0\] \
-inTR \
-signal GammaVar \
-noise Zero \
-bucket 7 dsc_gampara.nii.gz \
-brick 0 scoef 0 t0 \
-brick 1 scoef 1 amp \
-brick 2 scoef 2 rise \
-brick 3 scoef 3 decay \
-brick 4 fstat fstat \
-brick 5 area auc \
-brick 6 smax smax \
-sconstr 0 ${t1} ${t2} \
-sconstr 1 0 100  \
-sconstr 2 0 6  \
-sconstr 3 0 3 \
-snfit dsc_gamfit.nii.gz -jobs 32

# Generate all maps
# BAT
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[0\] -expr "a" -prefix bat.nii.gz

# RISE
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -expr "a" -prefix rise.nii.gz

# DECAY
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[3\] -expr "a" -prefix decay.nii.gz

# TTP
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -b dsc_gampara.nii.gz\[3\] -c bat.nii.gz -expr "a*b + c" -prefix ttp.nii.gz

# FSTAT (named SMS smoothness)
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[4\] -expr "a" -prefix sms.nii.gz

# AUC
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[5\] -expr "a" -prefix auc.nii.gz

# Max
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[6\] -expr "a" -prefix cmax.nii.gz

# FWHM: This is an approximation ... to be checked
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -b dsc_gampara.nii.gz\[3\] -expr "4*b*sqrt(a)" -prefix fwhm.nii.gz

# Cleaning up
rm dsc_gampara.nii.gz
