#!/bin/tcsh
cd $1

which 3dcalc >& /dev/null
if ($status == 0) then
    set pr = `which 3dcalc`
    set afni_gamvar = $pr:h
else
    set afni_gamvar = `dirname "$0"`
endif

dirname "$0"
set tr = `$afni_gamvar/3dinfo -TR conc.nii.gz`

# Find start of bolus, end of bolus and end of recirculation
$afni_gamvar/3dTstat -prefix conc_mean.nii.gz -mean conc.nii.gz
$afni_gamvar/3dmaskave -q -mask conc_mean.nii.gz conc.nii.gz > conc_ave.1D

# Bolus start time
set b = `$afni_gamvar/3dTstat -basepercent 15 -onset -prefix stdout: conc_ave.1D\'`

# Bolus end
set c = $2

3dTstat -max -prefix conc_max.nii.gz conc.nii.gz
set p = `3dmaskave -q -perc 99 conc_max.nii.gz`
set amp_max = `echo "10*$p" | bc -l`

# Start time of the gamma variate: 6 sec before $b
set t1 = `echo "${tr}*${b} - 6" | bc -l`
# Start time of the gamma variate: 6 sec after $b
set t2 = `echo "${tr}*${b} + 6" | bc -l`

# Need to be cleanup for it to continue
rm dsc_gamfit.nii.gz dsc_gampara.nii.gz

# Fit a gamma through the concentration
setenv AFNI_MODELPATH $afni_gamvar
$afni_gamvar/3dNLfim -input conc.nii.gz\[0..$c\] -ignore 0 -mask conc_mean.nii.gz \
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
-sconstr 1 0 $amp_max  \
-sconstr 2 0 6  \
-sconstr 3 0 3 \
-snfit dsc_gamfit.nii.gz -jobs 32



# BAT
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[0\] -expr "a" -prefix bat.nii.gz

# RISE
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -expr "a" -prefix rise.nii.gz

# DECAY
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[3\] -expr "a" -prefix decay.nii.gz

# TTP
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -b dsc_gampara.nii.gz\[3\] -c bat.nii.gz -expr "a*b" -prefix rttp.nii.gz
$afni_gamvar/3dcalc -float -a bat.nii.gz -b rttp.nii.gz -expr "a+b" -prefix ttp.nii.gz 

# FSTAT (named SMS smoothness in model-free)
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[4\] -expr "a" -prefix sms.nii.gz

# AUC
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[5\] -expr "a" -prefix auc.nii.gz

# Max
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[6\] -expr "a" -prefix cmax.nii.gz

# FWHM: This is an approximation ... to be checked
$afni_gamvar/3dcalc -float -a dsc_gampara.nii.gz\[2\] -b dsc_gampara.nii.gz\[3\] -expr "2.355*b*sqrt(a)" -prefix fwhm.nii.gz

# Generate the fit gamma variate as dsc_gamfit.nii.gz but with the tail
$afni_gamvar/3dcalc -float -g conc.nii.gz -r rise.nii.gz -d decay.nii.gz -a dsc_gampara.nii.gz\[1\] -c bat.nii.gz \
-expr "step(t-c)*a*exp(r*log(t-c))*exp(-(t-c)/d)" -prefix tmp_dsc_gamfit_full.nii.gz

# Threshold for crazy values
3dTstat -max -prefix tmp_gamma_max.nii.gz tmp_dsc_gamfit_full.nii.gz
3dcalc -a tmp_dsc_gamfit_full.nii.gz -b tmp_gamma_max.nii.gz -c conc_max.nii.gz -expr "a*(step(1.5*c-b))" -prefix dsc_gamfit_full.nii.gz

rm tmp_dsc_gamfit_full.nii.gz tmp_gamma_max.nii.gz  conc_max.nii.gz

# Cleaning up
#rm dsc_gampara.nii.gz
