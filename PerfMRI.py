#!/usr/bin/env python

# Import packages
from glob import glob
import os
import shutil
from tkinter import *
from tkinter import filedialog, simpledialog, ttk, messagebox

import subprocess
import threading
import numpy as np
trapz = getattr(np, "trapezoid", np.trapz)
import numpy.linalg as npl
import nibabel as nb
import nibabel.orientations as ornt
from nibabel.processing import resample_from_to

from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import Button as Button_mpl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import (LinearSegmentedColormap,ListedColormap, Normalize, BoundaryNorm)
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.path import Path

from reorient_nii import load as load_orient
from nilearn.masking import compute_epi_mask, apply_mask
from nilearn.image import smooth_img
import math
try:
    import pydicom
    pydicom.config.disable_enconders = True
    pydicom_available = True
except:
    pydicom_available = False
try:
    import dicom2nifti
    dicom2nifti_available = True
except:
    dicom2nifti_available = False

from scipy.signal import convolve, argrelmin, argrelmax, find_peaks
from scipy.linalg import toeplitz, circulant
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation



from sklearn.linear_model import LinearRegression
from functools import partial

from nipy.algorithms.registration.groupwise_registration import SpaceRealign
from nilearn.image import resample_to_img
from sklearn.cluster import KMeans
from tkinter import font
try:
    import ants
    ANTS_AVAILABLE = True
    import ants
except Exception:
    ANTS_AVAILABLE = False


window = Tk()


# Platform specific
import platform

if platform.system() == 'Windows':
    bt_small = 3
    bt_font = font.Font(size=8)
    font.nametofont("TkDefaultFont").configure(family="Arial", size=8)
    font.nametofont("TkTextFont").configure(family="Arial", size=10)
    font.nametofont("TkFixedFont").configure(family="Arial", size=8)

    metric_width = 60
    value_width = 40
    frame_ui_canvas_width = 400
    scroll_sensitivity = 120
    time_map_width = 22


elif platform.system() == 'Darwin':   # Mac
    bt_small = 1
    bt_font = font.Font(size=13)

    font.nametofont("TkFixedFont").configure(
        family=".AppleSystemUIFont",
        size=13
    )

    metric_width = 100
    value_width = 60
    frame_ui_canvas_width = 600
    scroll_sensitivity = 1
    time_map_width = 18


elif platform.system() == 'Linux':
    bt_small = 1
    bt_font = font.Font(size=11)

    font.nametofont("TkDefaultFont").configure(
        family="Ubuntu",
        size=10
    )

    font.nametofont("TkTextFont").configure(
        family="Ubuntu",
        size=10
    )

    font.nametofont("TkFixedFont").configure(
        family="Ubuntu",
        size=10
    )

    metric_width = 100
    value_width = 60
    frame_ui_canvas_width = 600
    scroll_sensitivity = 1
    time_map_width = 18

# Useful function for debugging
def np_info(array):
    if np.ma.isMaskedArray(array):
        mask_info = f"mask shape: {array.mask.shape}"
    else:
        mask_info = "mask: None"
    print("type:", type(array))
    print("shape:", array.shape)
    print("dtype:", array.dtype)
    print(mask_info)
    print("")

# Show/hide overlay ==================================================================================
# ====================================================================================================

def view_overlay():
    update_slice(slice_n.val)
    overlay_names = ['ove4', 'ove5', 'ove6', 'ove1b']
    visible = chk_map_state.get()
    for name in overlay_names + ['ove_roi']:
        if name in globals():
            print(name, id(globals()[name]))
    for name in overlay_names:
        if name in globals():
            globals()[name].set_visible(visible)
    
    canvas_fig.draw()

def view_func():
    if chk_func_state.get():
        update_slice(slice_n.val)
        ove1.set_visible(True)
    else:
        ove1.set_visible(False)
    canvas_fig.draw()

# Show/hide average brain time series ===============================================================
# ====================================================================================================

def view_ave_brain():
    if chk_ave_brain_state.get():
        line_brain_ave.set_visible(True)
        plt.draw()
    else:
        line_brain_ave.set_visible(False)
        plt.draw()

def view_aif():
    list_lines = [item[1] for item in ijk_lines]
    if chk_aif_state.get():
        for line in list_lines:
            line.set_visible(True)
        plt.draw()
    else:
        for line in list_lines:
            line.set_visible(False)
        plt.draw()

def view_ave_aif():
    if chk_ave_aif_state.get():
        line_aif_ave.set_visible(True)
        plt.draw()
    else:
        line_aif_ave.set_visible(False)
        plt.draw()

def autoscale_graph():
    if chk_autoscale_state.get():
        ax3.relim()        # Recompute limits based on all data (even those out of view)
        ax3.autoscale()
    else:
        # Freeze the scale by explicitly setting the limits:
        ax3.set_xlim(ax3.get_xlim())  # Set x-axis limits to the stored values
        ax3.set_ylim(ax3.get_ylim())  # Set y-axis limits to the stored values
    plt.draw()

def view_cross():
    state = chk_cross_state.get()
    for cross in (cross1, cross2, cross3, cross4):
        cross[0].set_visible(state)
        cross[1].set_visible(state)
    plt.draw()

# These are the name of images in top left corners like (TTP,MTT ... )
def view_label():
    state = chk_label_state.get()
    label1_text.set_visible(state)
    for txt in ax1.texts:
        txt.set_visible(state)
    for txt in ax4.texts:
        if txt.get_text() == label4:
            txt.set_visible(state)
    for txt in ax5.texts:
        if txt.get_text() == label5:
            txt.set_visible(state)
    for txt in ax6.texts:
        if txt.get_text() == label6:
            txt.set_visible(state)
    plt.draw()


# Info function
def write_info(textbox, textinfo, tag="info"):
    textbox.config(state="normal")
    textbox.insert("end", textinfo + "\n", tag)
    textbox.see("end")
    textbox.config(state="disabled")
    content = textbox.get("1.0", "end-1c") 
    p = os.path.join(dir_perfmri,'perfmri_processing.txt')
    with open(p, "w", encoding="utf-8") as f:
        f.write(content) 

def convert_func():
    dicom_dir = filedialog.askdirectory(title="Select Directory")
    output_dir = os.path.abspath(os.path.join(dicom_dir, os.pardir))
    os.chdir(output_dir)
    output_file = os.path.join(output_dir, 'func.nii.gz')
    dicom2nifti.dicom_series_to_nifti(dicom_dir, output_file=output_file, reorient_nifti=True)
    # Get the first DICOM file in the directory
    first_file = next(os.scandir(dicom_dir)).path
    # Read the DICOM file to get the TR
    dcm = pydicom.dcmread(first_file)
    correct_tr = dcm.RepetitionTime / 1000.0  # Convert to seconds if needed
    # Load the NIfTI file and modify the TR
    nifti_img = nb.load(output_file)
    header = nifti_img.header
    header['pixdim'][4] = correct_tr  # Set the correct TR in seconds
    # Save the modified NIfTI file if the TR was incorrect
    nb.save(nifti_img, output_file)
    lb_convert.config(fg="red")

#TR = nb_func.header['pixdim'][4]
# Convert files from dicom to nifti
# ANAT

def convert_anat():
    dicom_dir = filedialog.askdirectory(title="Select Directory")
    output_dir = os.path.abspath(os.path.join(dicom_dir, os.pardir))
    os.chdir(output_dir)
    output_file = os.path.join(output_dir, 'anat.nii.gz')
    dicom2nifti.dicom_series_to_nifti(dicom_dir, output_file=output_file, reorient_nifti=True)

maindir = '.'
def click_choose_funcfile():
    global funcfullfile, maindir
    funcfullfile = filedialog.askopenfilename(initialdir=maindir,title="Select Functional image (4D)",  filetypes=[("NIFTI files", '*.nii'),("Compressed NIFTI files",'*.gz'),("AFNI files",'*.HEAD')])
    funcfile = os.path.basename(funcfullfile)
    maindir = os.path.dirname(funcfullfile)
    os.chdir(maindir)
    funcdir.config(text='Functional  : ' + funcfullfile)
    window.focus_force()

def click_choose_anatfile():
    global anatfullfile, maindir
    anatfullfile = filedialog.askopenfilename(initialdir=maindir,title="Select Anatomical image (3D)",  filetypes=[("NIFTI files", '*.nii'),("Compressed NIFTI files",'*.gz'),("AFNI files",'*.HEAD')])
    anatfile = os.path.basename(anatfullfile)
    maindir = os.path.dirname(anatfullfile)
    os.chdir(maindir)
    anatdir.config(text='Anatomical  : ' + anatfullfile)
    window.focus_force()


def calculate_fov(nifti_image):
    """
    Calculate the Field of View (FOV) of a NIfTI image.
    
    Parameters:
    nifti_image (nib.Nifti1Image or nib.Nifti2Image): The loaded NIfTI image.

    Returns:
    list: A list representing the field of view in the format [xmin, xmax, ymin, ymax].
    """
    aff = nifti_image.affine
    shape = nifti_image.shape
    
    # Calculate the corners of the FOV
    xmin, ymin, z_slice, _ = -aff @ np.array([0, shape[1] - 1, 1, 1])  # Top left corner
    xmax, ymax, z_slice, _ = -aff @ np.array([shape[0] - 1, 0, 1, 1])  # Bottom right corner
    
    # Return FOV as [xmin, xmax, ymin, ymax]
    return [xmin, xmax, ymin, ymax]
 
def is_oblique(affine):
    return np.any(affine[:3, :3] != np.diag(np.diag(affine[:3, :3])))
        

def oblique_image_like(master_im, target_im):
    # Get target image's voxel size
    target_voxel_size = target_im.header.get_zooms()[:3]

    # Get master image's affine and shape
    affine_master = master_im.affine.copy()
    shape_master = np.array(master_im.shape)[:3]
    # Compute field of view of master image in mm
    fov_mm = shape_master * np.sqrt((affine_master[:3, :3] ** 2).sum(0))

    # Compute new shape for target in voxels, matching FOV but using target voxel size
    new_shape = np.ceil(fov_mm / target_voxel_size).astype(int)

    # Extract rotation from master affine
    U, _, Vt = np.linalg.svd(affine_master[:3, :3])
    R = U @ Vt  # rotation matrix (no scaling)

    # Build new affine: preserve master rotation, use target voxel size, preserve translation
    new_affine = np.eye(4)
    new_affine[:3, :3] = R * target_voxel_size
    new_affine[:3, 3] = affine_master[:3, 3]

    # Create a reference image with this affine and shape
    ref_img = nb.Nifti1Image(np.zeros(new_shape), new_affine)

    # Resample target image to this new space
    resampled_target = resample_from_to(target_im, ref_img)

    return resampled_target


def deoblique_affine(nii_img):
    # Get current affine and voxel size
    affine = nii_img.affine.copy()
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Build a new diagonal affine (no rotation), preserving translation
    new_affine = np.eye(4)
    new_affine[0, 0] = np.sign(affine[0, 0]) * voxel_sizes[0]
    new_affine[1, 1] = np.sign(affine[1, 1]) * voxel_sizes[1]
    new_affine[2, 2] = np.sign(affine[2, 2]) * voxel_sizes[2]
    new_affine[:3, 3] = affine[:3, 3]  # keep translation

    return nb.Nifti1Image(nii_img.get_fdata(), new_affine, header=nii_img.header)

def load_orient_nifti_afni(fullfile):
    # Check file extension and load the appropriate file format
    basename,file_extension = os.path.splitext(fullfile)
    
    if file_extension in ['.nii', '.gz']:  # NIfTI files
        nifti_image = load_orient(fullfile, "LPS")  # Reorient NIfTI image to LPS
        
    elif file_extension == '.HEAD':  # AFNI file (.HEAD)
        # Load the AFNI file
        afni_image = nb.load(fullfile)
        # Remove the extension
        rootname, _ = os.path.splitext(fullfile)  # Removes .HEAD, result is root+orig
        # Remove AFNI suffix like +orig or +tlrc
        rootname = rootname.split('+')[0]  # Extract root from root+orig
        nb.save(afni_image,rootname + '.nii.gz')
        #save2nifti(afni_image, afni_image.affine, 1, os.path.dirname(fullfile), rootname + '.nii.gz')
        nifti_image = load_orient(rootname + '.nii.gz', "LPS")  # Reorient NIfTI image to LPS
        # Squeeze the NIfTI image to remove singleton dimension if it exists
        if nifti_image.shape[-1] == 1:  # Check if the last dimension is 1 (indicating singleton)
            nifti_image = nifti_image.slicer[..., 0]  # Remove the singleton dimension        
    else:
        raise ValueError("Unsupported file format. Only NIfTI and AFNI (.HEAD) files are supported.")
    return nifti_image



def popup_wait():
    global popup
    # Create popup
    popup = Toplevel(window)
    popup.title("")
    Label(popup, text="⏳ Please wait...\n\n🔴 Processing ...",).pack(padx=20, pady=20)
    popup.geometry("200x100+500+300")
    popup.transient(window)   # keep on top of main window
    popup.grab_set()        # block interaction with main window

# Declaring bt_prepro_history to keep track of the preprocessing button used for the UNDO
bt_prepro_history = []

# Import NIFTI images and initial display
# ============================================================================================
def press_load_raw():
    popup_wait()
    threading.Thread(target=load_raw, daemon=True).start()
    

#@profile
def load_raw():
    global anat,func_sc, func_mean, aif, data1, label1_text, brain_ave
    global aff_func_orig, form_code_func_orig
    global func_mask, func_mask_4d,func_masked
    global nb_anat, nb_func
    global t, TR
    global aff_anat,aff_func
    global fov_anat, fov_func
    global und1,und4,und5,und6
    global ove1,ove1b,ove_roi
    global I0,J0,K0,i0,j0,k0
    global min_anat, max_anat
    global text_coord
    global dir_perfmri,dir_preprocess,dir_relative_modfree, dir_relative_gamvar
    global dir_quantitative_decon, dir_quantitative_expon
    global dir_cvr, dir_cvr_lag, dir_cvr_rt, dir_cvr_svd, dir_cvr_expon
    global roi_map, ove_roi

    # Clean up all previous data
    data_vars = [key for key in globals() if not key.startswith("__") and key not in interface_vars and key != 'interface_vars']
    for key in data_vars:
        if (key != 'anatfullfile'
            and key != 'funcfullfile'
            and key != 'slice_n'
            and key != 'nb_func_orig'
            and key != 'popup'
        ):
            del globals()[key]
    #================================================================
    # A counter for saving the pre-processing steps
    global pr
    pr = 0

    # Declaring bt_prepro_history to keep track of the preprocessing button used for the UNDO
    for bt in bt_prepro_history:
        bt.config(fg="black")
    

    # Make all the appropriate sub-directories
    dir_perfmri = os.path.join(maindir,'PerfMRI')
    dir_preprocess  = os.path.join(dir_perfmri,'preprocess')
    if os.path.exists(dir_preprocess):
        shutil.rmtree(dir_preprocess)  # Deletes the entire folder and its contents
        os.makedirs(dir_preprocess)  
    os.makedirs(dir_preprocess, exist_ok=True)
    dir_relative_modfree  = os.path.join(dir_perfmri, 'relative_modfree')
    dir_relative_gamvar = os.path.join(dir_perfmri, 'relative_gamvar')
    dir_quantitative_expon  = os.path.join(dir_perfmri, 'quantitative_expon')
    dir_quantitative_decon  = os.path.join(dir_perfmri, 'quantitative_decon')
    dir_cvr  = os.path.join(dir_perfmri,'cvr_analysis')
    dir_cvr_lag = os.path.join(dir_perfmri,'cvr_lag_analysis')
    dir_cvr_rt = os.path.join(dir_perfmri,'cvr_rt_analysis')
    dir_cvr_svd = os.path.join(dir_perfmri,'cvr_quantitative_decon')
    dir_cvr_expon =os.path.join(dir_perfmri,'cvr_quantitative_expon')


    # Load the current options of the file into the text widget
    filename = os.path.join(dir_perfmri, 'PerfMRI_advanced_options.txt')
    if not os.path.exists(filename):
        set_advanced_options()
        
    with open(filename, 'r') as file:
        content = file.read()
        text_widget.delete('1.0', END) 
        text_widget.insert('1.0', content)

    # Loading datasets
    nb_func_orig = load_orient_nifti_afni(funcfullfile)
    aff_func_orig = nb_func_orig.affine
    #form_code_func_orig = nb_func_orig.header['sform_code']
    form_code_func_orig = 1

    # Deoblique dataset
    nb_func = deoblique_affine(nb_func_orig)
    data1 = nb_func.get_fdata().transpose(1,0,2,3)
    data1 = np.ma.masked_array(data1, mask=np.zeros_like(data1, dtype=bool)) #changeit
    
    # Extract affine and FOV from the deobliqued header
    aff_func = nb_func.affine
    fov_func = calculate_fov(nb_func)
    
    

    # Make the mean and calculate min_func, max_func for display
    func_mean = np.ma.mean(data1,axis=3)
    min_func, max_func = vminvmax_percentile(func_mean,3,3)

    # Initialize the aif array and func_mask
    aif = np.ma.masked_all_like(func_mean)
    roi_map = np.ma.masked_all_like(func_mean)
    func_mask = np.ones_like(np.mean(data1,axis=3),dtype=np.float64)
    func_masked = np.ma.array(data1)
    

    
    # Import anat and calculate FOV
    # If user does not have an anatomical, use the mean func as anatomical
    if 'anatfullfile' in globals():
        nb_anat_noreg = load_orient_nifti_afni(anatfullfile)

        if is_oblique(aff_func_orig) or is_oblique(nb_anat_noreg.affine):
            nb_anat_reg = oblique_image_like(nb_func_orig,nb_anat_noreg)
            nb_anat = deoblique_affine(nb_anat_reg)
            anat = nb_anat.get_fdata()
            if anat.ndim == 4:
                anat = np.mean(anat,axis=3).transpose(1,0,2)
            else:
                anat = anat.transpose(1,0,2)
            save2nifti(anat,nb_anat.affine,2,dir_preprocess, 'anat2func_obl.nii.gz')
        else:
            nb_anat = nb_anat_noreg
            anat = nb_anat.get_fdata()
            if anat.ndim == 4:
                anat = np.mean(anat,axis=3).transpose(1,0,2)
            else:
                anat = anat.transpose(1,0,2)
        

    else:
        # Use the mean functional as the underlay
        nb_anat = nb_func
        anat = func_mean.copy()

    # Extract affine and FOV from anat
    aff_anat = nb_anat.affine
    fov_anat = calculate_fov(nb_anat)
    
    # Calculate min_anat, max_anat for color scale
    min_anat, max_anat = vminvmax_percentile(anat,3,3)

    # Get the label for the raw functional image displayed in axs1
    basename, _ = os.path.basename(funcfullfile).split(".",1)
    label1 = basename.upper()
    label1 = f'func_{pr:02d}_raw'
    label1 = label1.upper()

    # Calculate the approximative center for first display
    I0, J0, K0 = np.array(nb_func.shape[0:3]) // 2
    i0,j0,k0 = ijk_func2anat([I0,J0,K0])

    global slice_n
    # Remove the existing slider's axis if it already exists
    if "slice_n" in globals():
        slice_n.valtext.remove()  # remove the old text
        slice_n.label.remove()
        slice_n.ax.remove()
        del slice_n
    
    # Now create a new slider
    slider_pos = fig.add_axes([0.97, 0.03, 0.03, 1])
    slice_n = Slider(slider_pos, "Slices", 0, anat.shape[2]-1, valinit=k0, valstep=1,
                    color='darkgrey', orientation='vertical')
    slice_n.valtext.set_color('red')
    slice_n.valtext.set_fontsize(16)
    slice_n.valtext.set_position((-1, 0))
    slice_n.on_changed(update_slice)

    def on_scroll(event):
        if event.inaxes in [ax1, ax3, ax4, ax5, ax6, slice_n.ax]:
            s = 1 if event.step < 0 else -1  # Force ±1 step
            new_val = slice_n.val + s
            new_val = max(slice_n.valmin, min(slice_n.valmax, new_val))
            slice_n.set_val(new_val)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Initial displayed slice and time series
    # ===============================================================================================

    xborder = 0.05*(fov_anat[1]-fov_anat[0])
    for ax in [ax1, ax5 ,ax6 ,ax4]:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([fov_anat[0], fov_anat[1] + xborder])

    # Calculate font size for label
    h_box = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height

    ove1 = ax1.imshow(data1[:, :, K0,0],cmap='gist_heat',alpha=0.7,extent=fov_func,zorder=2,vmin=min_func,vmax=max_func)
    ove1b = ax1.imshow(aif[:, :, K0],cmap='spring',alpha=1,extent=fov_func,zorder=2)
    label1_text = ax1.text(0.05, 0.95,label1, color='white', fontsize=2.5*h_box, ha='left', va='top', transform=ax1.transAxes)
    ax1.text(0.05, 0.05, 'R', color='white', fontsize=2.5*h_box, ha='left', va='bottom', transform=ax1.transAxes)
    ax1.text(0.95, 0.05, 'L', color='white', fontsize=2.5*h_box, ha='right', va='bottom', transform=ax1.transAxes)
    und1 = ax1.imshow(anat[:, :, k0],cmap='gray',extent=fov_anat,zorder=1,vmin=min_anat,vmax=max_anat)
    ove_roi = ax4.imshow(roi_map[:, :, K0],cmap=cmap_roi,alpha=1,extent=fov_func,zorder=2,vmin=0.5,vmax=6.5)
    und4 = ax4.imshow(anat[:, :, k0],cmap='gray',extent=fov_anat,zorder=1,vmin=min_anat,vmax=max_anat)
    und5 = ax5.imshow(anat[:, :, k0],cmap='gray',extent=fov_anat,zorder=1,vmin=min_anat,vmax=max_anat)
    und6 = ax6.imshow(anat[:, :, k0],cmap='gray',extent=fov_anat,zorder=1,vmin=min_anat,vmax=max_anat)
    
    

    global cross1, cross2, cross3, cross4
    cross1 = plot_cross(ax1,0,0)
    cross2 = plot_cross(ax5,0,0)
    cross3 = plot_cross(ax6,0,0)
    cross4 = plot_cross(ax4,0,0)

    # Plot the time series and average brain
    TR = nb_func.header['pixdim'][4]
    # Re-display all
    resetplot(ax3,data1,TR)
   
    # Set some defaults or suggested values
    sb_fwhm_t.delete(0,END)
    sb_fwhm.delete(0,END)
    vox_size = np.mean(nb_func.header['pixdim'][1:4])
    sb_fwhm_t.insert(0,np.round(1.5*TR,1))
    sb_fwhm.insert(0,np.round(1.5*vox_size,1))

    # Fill out the info box with Image parameters 
    info_txt.config(state="normal")
    info_txt.delete("1.0","end") # Clean the info text box
    write_info(info_txt,"Functional scan:",tag="title")
    write_info(info_txt,f"Voxel size = {np.round(nb_func.header['pixdim'][1:4],2)} mm")
    write_info(info_txt, f"Matrix = {nb_func.header['dim'][1:4]}")
    write_info(info_txt, f"#Volumes = {data1.shape[-1]}")
    write_info(info_txt, f"TR = {TR} s")
    write_info(info_txt, f"descrip = {nb_func.header['descrip']}")
    write_info(info_txt,"\n")
    write_info(info_txt,"Anatomical scan:",tag="title")
    write_info(info_txt,f"Voxel size = {np.round(nb_anat.header['pixdim'][1:4],2)} mm")
    write_info(info_txt, f"Matrix = {nb_anat.header['dim'][1:4]}")
    write_info(info_txt, f"descrip = {nb_anat.header['descrip']}")             
    write_info(info_txt,"\n")
    write_info(info_txt,"Pre-processing:",tag="title")

    plt.draw()
    window.after(0, popup.destroy)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_raw.nii.gz')


def set_vlines():
    # Set lines to trim signal.  If dataset is less than 120s, it is likely a DSC file
    # in which case a quick calculation of end of bolus is done.  
    if t[-1] < 120:
        conc = -1*(np.log(data1 / data1[...,[2]]))
        mask = np.isnan(conc) | np.isinf(conc)
        conc = np.ma.array(conc, mask=mask)
        conc_ave = np.ma.mean(conc, axis=(0, 1, 2))
        i_end_base, i_end_bolus = find_bolus_lines(conc_ave,TR)
        vline1.set_xdata([TR*2,TR*2])
        vline2.set_xdata([TR*i_end_bolus,TR*i_end_bolus])
    else:
        vline1.set_xdata([0,0])
        vline2.set_xdata([t[-1],t[-1]])
    plt.draw()

def press_trim_signal():
    popup_wait()
    threading.Thread(target=trim_signal, daemon=True).start()


def trim_signal():   
    global data1, pr, i_start, i_end_bolus
    pr += 1
    v1 = round(vline1.get_xdata()[0]/TR)
    v2 = round(vline2.get_xdata()[0]/TR)
    if v1 > v2:
        i_start = v2 ; i_end_bolus = v1
    else:
        i_start = v1 ; i_end_bolus = v2

    i_start = np.max(i_start,0)
    i_end_bolus = np.min([i_end_bolus,data1.shape[-1]])
    n_end_vol = data1.shape[-1] - i_end_bolus - 1
    data1 = data1[...,i_start:i_end_bolus+1]
    

    # Re-display all
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_TRI')
    
    window.after(0, popup.destroy)
    # Now place again the vlines
    vline1.set_xdata([0,0])
    vline2.set_xdata([2*TR,2*TR]) 
    plt.draw()  
    window.after(0, popup.destroy)
    bnt_trim_signal.config(fg="red")
    bt_prepro_history.append(bnt_trim_signal)
    save2nifti(data1,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_tri.nii.gz')
    write_info(info_txt, f"Trim Signal: Removed {i_start} vol from start, {n_end_vol} from end")  


def on_slicetime_change(*args):
    global slicetime_acq_file
    if tkvar_slicetime.get() == "Text file...":
        slicetime_acq_file = filedialog.askopenfilename(initialdir=maindir,title="Select timing file...")
        window.focus_force()

# Time Realign  / Slice time correction
def press_time_realign():
    popup_wait()
    threading.Thread(target=time_realign, daemon=True).start()

def time_realign():
    global data1, pr, bt_prepro_current
    pr += 1

    if tkvar_slicetime.get() == "Text file...":
        sliceAcquisitionTime = np.loadtxt(slicetime_acq_file)
        data1 = time_realign_calc(data1,sliceAcquisitionTime,TR)

    if tkvar_slicetime.get() == "Sequential+":
        nsl = data1.shape[2]
        sliceAcquisitionTime = np.arange(0, TR, TR/nsl)
        data1 = time_realign_calc(data1,sliceAcquisitionTime,TR)
   

    if tkvar_slicetime.get() == "Alternative+":
        nsl = data1.shape[2]
        sliceAcquisitionTime = mri_interleave(nsl,TR)
        data1 = time_realign_calc(data1,sliceAcquisitionTime,TR)

    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_TRE')

    plt.draw()
    window.after(0, popup.destroy)
    bt_slicetime.config(fg="red")
    bt_prepro_history.append(bt_slicetime)
    save2nifti(data1,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_tre.nii.gz')
    write_info(info_txt, f"Slice Time corrected")

def mri_interleave(nz,TR):
    dt = TR/nz 
    seqplus = np.arange(0,TR,dt)
    if (nz % 2) == 1:
        seqplus = np.append(seqplus,seqplus[-1]+dt)
    first_half = seqplus[0:len(seqplus)//2]
    second_half = seqplus[len(seqplus)//2:]
    interleaved = np.ravel(np.column_stack((first_half, second_half)))
    interleaved = interleaved[0:nz]
    return interleaved

def time_realign_calc(fMRI, sliceAcquisitionTime, TR):
    ni, nj, nk, nt = fMRI.shape
    fMRI_shifted = np.ma.zeros_like(fMRI)
    expected_times = np.arange(0, nt * TR, TR)  # 1D array
    opts = read_advanced_options()
    slice_time_interp = opts['slice_time_interp']

    for k in range(nk):
        real_times = expected_times + sliceAcquisitionTime[k]  
        # Interpolation along axis=2 (time axis)
        fs = interp1d(real_times, fMRI[:, :, k, :], kind=slice_time_interp, bounds_error=False,
                      fill_value=(fMRI[:, :, k, 0], fMRI[:, :, k, -1]), axis=2)
        fMRI_shifted[:, :, k, :] = fs(expected_times)  
    return fMRI_shifted


# def on_correct_time_change(*args):
#     global slicetime_acq_file
#     if tkvar_correct_time.get() == "Text file...":
#         slicetime_acq_file = filedialog.askopenfilename(initialdir=maindir,title="Select timing file...")
#         window.focus_force()

# def press_correct_time():
#     global data1, pr
#     global data5,ove5,und5,mapval5,label5
#     global data6,ove6,und6,mapval6,label6
#     pr += 1


#     name_t_metrics = ['bat','ttp','rttp','fm']
#     t_metrics = []
#     for m in name_t_metrics:
#         _, tmp, affine = load_nifti(dir_relative_modfree, m+'.nii.gz')
#         t_metrics.append(tmp)
#     t_metrics = np.stack(t_metrics, axis=-1)  # (Nx, Ny, Nz, Nmetrics)


#     if tkvar_correct_time.get() == "Text file...":
#         sliceAcquisitionTime = np.loadtxt(slicetime_acq_file)
#         sliceAcquisitionTime = sliceAcquisitionTime.reshape(1, 1, -1, 1)
#         t_metrics_corrected = t_metrics + sliceAcquisitionTime

#     if tkvar_correct_time.get() == "Sequential+":
#         nsl = data1.shape[2]
#         sliceAcquisitionTime = np.arange(0, TR, TR/nsl)
#         sliceAcquisitionTime = sliceAcquisitionTime.reshape(1, 1, -1, 1)
#         t_metrics_corrected = t_metrics + sliceAcquisitionTime
   
#     if tkvar_correct_time.get() == "Alternative+":
#         nsl = data1.shape[2]
#         sliceAcquisitionTime = mri_interleave(nsl,TR)
#         sliceAcquisitionTime = sliceAcquisitionTime.reshape(1, 1, -1, 1)
#         t_metrics_corrected = t_metrics + sliceAcquisitionTime

#     for i, m in enumerate(name_t_metrics):
#         corrected_metric = t_metrics_corrected[..., i]  # (Nx, Ny, Nz)
#         corrected_metric = np.ma.masked_where(func_mask==0,corrected_metric)
#         save2nifti(corrected_metric,aff_func_orig,form_code_func_orig,dir_relative_modfree,f"{m}_stc.nii.gz")

#     ttp_corr = np.ma.masked_where(func_mask==0,t_metrics_corrected[...,1])
#     label5 = 'TTP_STC'
#     data5,ove5,und5,mapval5 = show_map(ax5,ttp_corr,label5,'viridis',5,5,anat) 
    
#     bat_corr = np.ma.masked_where(func_mask==0,t_metrics_corrected[...,0])
#     label6 = 'BAT_STC'
#     data6,ove6,und6,mapval6 = show_map(ax6,bat_corr,label6,'viridis',5,5,anat) 
    
#     plt.draw()

def press_space_realign():
    popup_wait()
    opts = read_advanced_options()
    if opts['reg_afni'] == "yes":
        threading.Thread(target=afni_space_realign, daemon=True).start()
    else:
        threading.Thread(target=nipy_space_realign, daemon=True).start()
    


def afni_space_realign():
    """
    Run AFNI 3dvolreg and return motion parameters as numpy array.
    """
    global data1, pr
    pr += 1
    save2nifti(data1,aff_func_orig,form_code_func_orig, dir_preprocess, 'tobe_reg.nii.gz')
    infile = os.path.join(dir_preprocess,'tobe_reg.nii.gz')
    outfile = os.path.join(dir_preprocess,f'func_{pr:02d}_vre.nii.gz')
    param_file = os.path.join(dir_preprocess,'param_mot.1D')
    opts = read_advanced_options()
    moco_ref = opts['moco_ref']
    
    cmd = [
        "3dvolreg",
        "-clipit",
        "-base", moco_ref,
        "-prefix", outfile,
        "-1Dfile", param_file,
        infile
    ]

    # Run AFNI
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nb_data , data, label = load_nifti(dir_preprocess,f'func_{pr:02d}_vre.nii.gz')
    data1 = np.ma.masked_array(data,mask=data1.mask)
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_VRE')
    os.remove(infile)
    write_info(info_txt, f"Volume re-registration using AFNI 3dvolreg")
    window.after(0, popup.destroy)
    bt_space_realign.config(fg="red")
    bt_prepro_history.append(bt_space_realign)

def nipy_space_realign():
    global data1, pr
    pr += 1
    opts = read_advanced_options()
    moco_ref = int(opts['moco_ref'])

    nb_data1 = nb.Nifti1Image(data1,aff_func)
    realigner = SpaceRealign(images=nb_data1)
    realigner.estimate(
        refscan=moco_ref, 
        speedup=8,         # Higher value for faster but lower resolution alignment
        stepsize=1e-4,     # Larger step size for faster convergence
        optimizer='powell', # Tried a faster optimizer (Powell method)
        gtol=1e-5,
        maxiter=23         # Reduce the maximum number of iterations
    )

    nb_data_reg = realigner.resample(0) #changeit
    data1 = np.ma.masked_array(nb_data_reg.get_fdata(),mask=data1.mask)  # Extract data 

    # Re-display all
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_VRE')

    regs = realigner._transforms
    params = []  # start with a list
    # assuming tr[0] contains all volumes for your run
    for reg in regs[0]:
        # concatenate translation and rotation
        line = np.concatenate([np.degrees(reg.rotation),reg.translation])
        params.append(line)

    # convert to numpy array
    params = np.vstack(params)  # shape = (n_volumes, 6)
    np.savetxt(os.path.join(dir_preprocess,"param_mot.1D"), params, fmt="%.6f")

    write_info(info_txt, f"Volume re-registration using NIPY")
    window.after(0, popup.destroy)
    bt_space_realign.config(fg="red")
    bt_prepro_history.append(bt_space_realign)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_vre.nii.gz')
   

def press_view_motion():
    # Create a new top-level window
    mot_window = Toplevel()
    mot_window.title("Motion Parameters")
    mot_window.configure(bg='black')  # optional

    # Load motion parameters
    mot_param = np.loadtxt(os.path.join(dir_preprocess, "param_mot.1D"))
    Trep = globals().get("TR", 1)
    ti = Trep * np.arange(mot_param.shape[0])

    # Create the figure (no plt.figure)
    motfig = Figure(figsize=(14, 7), facecolor='black')
    ax_mot = motfig.add_subplot(111)
    ax_mot.set_facecolor('black')
    ax_mot.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    #for spine in ax_mot.spines.values():
    #    spine.set_visible(False)

    ax_mot.set_ylim(-2, 2)
    ax_mot.plot(ti, mot_param)
    ax_mot.set_title("Motion Parameters", color='white')
    ax_mot.set_xlabel(f"Time (s) with TR = {Trep}s", color='white')
    ax_mot.tick_params(colors='white')

    motfig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Embed the figure in the Toplevel
    canvas = FigureCanvasTkAgg(motfig, master=mot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Optional: handle closing of Toplevel window
    def on_close():
        canvas.get_tk_widget().destroy()
        mot_window.destroy()
    mot_window.protocol("WM_DELETE_WINDOW", on_close)

def press_reg_anat2func():
    popup_wait()
    threading.Thread(target=reg_anat2func, daemon=True).start()

import regtricks as rt
def reg_anat2func():
    save2nifti(anat,nb_anat.affine,2,dir_preprocess, 'tmp_anat.nii.gz')
    save2nifti(data1,aff_func_orig,2,dir_preprocess, 'tmp_func.nii.gz')
    anat2func = rt.Registration.from_flirt('anat2func.mat', src='tmp_anat.nii.gz', ref='tmp_func.nii.gz')
    anat = anat2func.apply_to_image('anat.nii.gz', ref='anat.nii.gz')
    save2nifti(anat,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_msd.nii.gz')

def press_mask_brain():
    popup_wait()
    threading.Thread(target=mask_brain, daemon=True).start()

def mask_brain():
    global func_mask,func_masked, data1, pr
    pr += 1

    if tkvar_masktype.get() == 'Automask':
        # Mask data1 outside the brain.
        nb_func_mask = compute_epi_mask(nb_func,lower_cutoff=0.1,upper_cutoff=0.9,exclude_zeros=False,opening=False)
        func_mask = nb_func_mask.get_fdata().transpose(1,0,2)
    
        #Dilate anatomical mask
        opts = read_advanced_options()
        d = int(opts['dil_mask'])
        if d > 0:
            func_mask = binary_dilation(func_mask, iterations=d).astype(int)

        # Now enforce functional constraint: all timepoints > 0.1
        func_mask = np.ma.masked_array(func_mask * np.all(data1 > 0.1, axis=3),
            mask=np.zeros(func_mask.shape, dtype=bool)
        )
        func_mask_4d = np.repeat(func_mask[..., np.newaxis], data1.shape[3], axis=3)
        data1 = np.ma.masked_where(func_mask_4d == 0, data1)
        

    elif tkvar_masktype.get() == 'Use Zeros':
        func_mask = np.mean(data1, axis=3)
        func_mask = (func_mask != 0).astype(float)
        func_mask_4d = np.repeat(func_mask[..., np.newaxis], data1.shape[3], axis=3)
        data1 = np.ma.masked_where(func_mask_4d == 0,data1)
    

    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_MSD')
    window.after(0, popup.destroy)
    bnt_mask_brain.config(fg="red")
    bt_prepro_history.append(bnt_mask_brain)

    # Saving the masked func data
    save2nifti(data1,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_msd.nii.gz')

    # Saving the 3D mask
    pr += 1
    save2nifti(func_mask,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_msk.nii.gz')
    write_info(info_txt, f"Brain masked")


def press_detrend_signal():
    popup_wait()
    threading.Thread(target=detrend_signal, daemon=True).start()

def detrend_signal():
    """
    Detrend the signal in the 4D array by removing a polynomial trend up to degree `pol_deg`.
    """
    global data1, pr
    pr += 1
    p = int(tkvar_poly.get())
    data1 = detrend_signal_calc(data1,TR,p)
    # Re-display all
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_DTR')
    window.after(0, popup.destroy)
    bt_detrend.config(fg="red")
    bt_prepro_history.append(bt_detrend)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_dtr.nii.gz')
    write_info(info_txt, f"Signal detrended (polynomial degree = {p})")
    
def detrend_signal_calc(array_4d, TR, pol_deg):
    """
    Detrend the signal in a 4D array by removing a polynomial trend up to degree `pol_deg`.
    
    Parameters:
    - array_4d: np.ndarray
        4D array (ni, nj, nk, nt) representing the brain image over time.
    - TR: float
        Repetition time (time between consecutive scans).
    - mask: np.ndarray
        3D mask array (ni, nj, nk), where non-zero values indicate voxels to detrend.
    - pol_deg: int, optional (default=1)
        Degree of the polynomial trend to remove (0 = constant, 1 = linear, etc.).

    Returns:
    - array_4d_detrended: np.ndarray
        Detrended 4D array, with the same shape as `array_4d`.
    """
    # Get the dimensions of the 4D array
    ni, nj, nk, nt = array_4d.shape
    
    # Flatten mask and find indices of non-zero voxels
    mask_flat = ~np.ma.getmaskarray(array_4d[...,-1])
    num_voxels = np.sum(mask_flat)
    
    # Reshape the 4D array into a 2D array (num_voxels, nt) for regression
    Y = array_4d[mask_flat, :]  # Only the masked voxels, shape: (num_voxels, nt)

    # Time vector based on TR
    t = TR * np.arange(nt)

    # Construct the polynomial design matrix X
    X = np.ones((nt, 1))  # Start with constant term (pol_deg=0)
    for p in range(1, pol_deg + 1):
        X = np.column_stack((X, t**p))  # Add higher polynomial terms

    # Perform the linear regression for all voxels
    model = LinearRegression(fit_intercept=False, n_jobs=-1)  # No intercept; X already has it
    model.fit(X, Y.T)  # Fit to each voxel (Y.T has shape (nt, num_voxels))
    
    # Predict the trend and detrend the signal
    Y_pred = model.predict(X)
    
    # Subtract only the higher-order polynomial terms (p >= 1)
    Y_detrended = Y - (Y_pred.T - Y_pred.T[:, [0]])  # Add back the constant term
    array_4d_detrended = np.zeros_like(array_4d)
    array_4d_detrended[mask_flat, :] = Y_detrended  # Fill in the detrended values
    return array_4d_detrended


def press_scale_signal():
    popup_wait()
    threading.Thread(target=scale_signal, daemon=True).start()

def scale_signal():
    global data1,nt,t,line_brain_ave,line_aif_ave, pr,i_start_base, i_end_base
    pr += 1
    
    v1 = round(vline1.get_xdata()[0]/TR)
    v2 = round(vline2.get_xdata()[0]/TR)
    if v1 > v2:
        i1 = v2 ; i2 = v1
    else:
        i1 = v1 ; i2 = v2
    
    i1 = np.max(i1,0)
    i2 = np.max([i2,0])
    
    if tkvar_imtype.get()=="Concentration":
        S = data1.copy()
        Spre = np.ma.mean(S[...,i1:i2+1],axis=3)
        save2nifti(Spre,aff_func_orig,form_code_func_orig,dir_perfmri,'Sbaseline.nii.gz')
        Ssc = S / Spre[..., np.newaxis]
        data1 = -(np.ma.log(Ssc))
        ax3_ylabel = "Concentration"
        write_info(info_txt, f"Signal scale: C = -log(S/S0) , S0 = mean(S[{i1}:{i2}])")

    
    elif tkvar_imtype.get()=="% baseline":
        # Scale to baseline
        data1_base = np.ma.mean(data1[:,:,:,i1:i2+1],axis=3)
        data1_sc_base = data1 / data1_base[..., np.newaxis]
        data1 = 100*(data1_sc_base - 1)
        np_info(data1)
        ax3_ylabel = "BOLD[%]"
        write_info(info_txt, f"Signal scale: Sc = 100x(S-S0)/S0 , S0 = mean(S[{i1}:{i2}])")
        
    elif tkvar_imtype.get()=="% mean":
        # Scale to the mean
        data1_mean = np.ma.mean(data1,axis=3)
        data1_sc_mean = data1 / data1_mean[..., np.newaxis]
        data1 = 100*(data1_sc_mean - 1)
        np_info(data1)
        ax3_ylabel = "BOLD[%]"
        write_info(info_txt, f"Signal scale: Sc = 100x(S-S0)/S0 , S0 = mean(S)")

    elif tkvar_imtype.get()=="% Percentile":
        # Scale to the mean
        opts = read_advanced_options()
        p = float(opts['scale_percentile'])
        data1_mean = np.nanpercentile(data1.filled(np.nan),p,axis=3)
        data1_sc_mean = data1 / data1_mean[..., np.newaxis]
        data1 = 100*(data1_sc_mean - 1)
        ax3_ylabel = "BOLD[%]"
        write_info(info_txt, f"Signal scale: Sc = 100x(S-S0)/S0 , S0: P-th percentile")
    
    # Plot the time series and average brain
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_SCL')
    # Place again the vlines in a sensible place
    vline1.set_xdata([0,0])
    vline2.set_xdata([t[1],t[1]])
    ax3.text(0.05, 0.95,ax3_ylabel, color='white',transform=ax3.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='left')
    window.after(0, popup.destroy)
    bt_scale_signal.config(fg="red")
    bt_prepro_history.append(bt_scale_signal)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_scl.nii.gz')


def press_spatial_smoothing():
    popup_wait()
    threading.Thread(target=spatial_smoothing, daemon=True).start()



def spatial_smoothing():
    global data1, pr
    pr += 1
    
    # Parameters for Gaussian smoothing
    fwhm = float(sb_fwhm.get())
    fwhm_x = (1/2.333) * fwhm / nb_func.header['pixdim'][1]
    fwhm_y = (1/2.333) * fwhm / nb_func.header['pixdim'][2]
    fwhm_z = (1/2.333) * fwhm / nb_func.header['pixdim'][3]

    smoothed_data = gaussian_filter(np.ma.filled(data1,0),sigma = (fwhm_x,fwhm_y,fwhm_z,0))
    
    
    mask4d = (~data1.mask).astype(float)

    smoothed_mask = gaussian_filter(mask4d,sigma = (fwhm_x,fwhm_y,fwhm_z,0)) 
    data1 = np.ma.array(smoothed_data / smoothed_mask,mask=data1.mask)
    
    # Plot the time series and average brain
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_SSM')
    window.after(0, popup.destroy)
    bt_spatial_smoothing.config(fg="red")
    bt_prepro_history.append(bt_spatial_smoothing)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_ssm.nii.gz')
    write_info(info_txt, f"Spatial smoothing: FWHM = {fwhm} mm ")

def press_temporal_smoothing():
    popup_wait()
    threading.Thread(target=temporal_smoothing, daemon=True).start()

def temporal_smoothing():
    global data1, pr
    pr += 1
    # Parameters for Gaussian smoothing
    fwhm_t = float(sb_fwhm_t.get())
    fwhm_t_sig = (1/2.333) * fwhm_t / TR

    smoothed_data = gaussian_filter(data1,sigma = (0,0,0,fwhm_t_sig))
    data1 = np.ma.array(smoothed_data,mask=data1.mask)
    
    # Plot the time series and average brain
    resetplot(ax3,data1,TR)
    label1_text.set_text(f'FUNC_{pr:02d}_TSM')
    window.after(0, popup.destroy)
    bt_temporal_smoothing.config(fg="red")
    bt_prepro_history.append(bt_temporal_smoothing)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_tsm.nii.gz')
    write_info(info_txt, f"Temporal smoothing: FWHM = {fwhm_t} s ")

# Undo last prepro step
def undo_prepro():
    global pr, data1, label1, func_mask
    if bt_prepro_history:
        bt = bt_prepro_history.pop()
        bt.config(fg="black", font=bt_font)
    if pr > 0:
        pattern = os.path.join(dir_preprocess, f"func_{pr:02d}*.nii.gz")
        file = glob(pattern)
        os.remove(file[0])
        if "_msk.nii.gz" in file[0]:
            pr-=1
            pattern2 = os.path.join(dir_preprocess, f"func_{pr:02d}_msd.nii.gz")
            file2 = glob(pattern2)
            os.remove(file2[0])
            if 'func_mask' in globals():
                global func_mask
                del func_mask
                func_mask = np.ones_like(np.mean(data1,axis=3),dtype=np.float64)
        
        pp = pr - 1
        pat = os.path.join(dir_preprocess, f"func_{pp:02d}*.nii.gz")
        fil = glob(pat)

        if "_msk.nii.gz" in fil[0]:
            pr -= 1
            _, m, _ = load_nifti(dir_preprocess, fil[0])

            func_mask = np.ma.masked_array(
                func_mask,
                mask=(m == 0)
            )

        info_txt.config(state="normal")
        info_txt.delete("end-2l", "end-1c")
        info_txt.config(state="disabled")
        pr-= 1
        pattern = os.path.join(dir_preprocess, f"func_{pr:02d}*.nii.gz")
        file = glob(pattern)
        _, data1, label1 = load_nifti(dir_preprocess,file[0])
        data1 = mask_4d_from_3d(func_mask==0,data1)
        resetplot(ax3,data1,TR)
        label1_text.set_text(label1)
        plt.draw()



# Make the xticks and yticks inside the graph and set their color to none.
def on_move_time_set(ax):
    xmin,xmax = ax.get_xlim()
    x_init = (xmin+xmax)/2
    vline = ax.axvline(x=x_init, color='blue', linestyle='-')

    # Text to display the x value
    text = ax.text(0.6, 0.8, f'time=--', transform=ax.transAxes, ha='left', va='top', color='white')

    def on_move_time(event):
        if event.inaxes == ax:
            vline.set_xdata([event.xdata,event.xdata])
            time_sec = event.xdata
            time_tr = int(time_sec//TR)
            text.set_text(f'time={time_tr} --> {time_sec:.2f}s')
            _, _, K = ijk_anat2func([0, 0, slice_n.val])
            ove1.set_data(data1[:,:,K,int(event.xdata//TR)])
            plt.draw()

    # Connect the motion event to the on_move function
    cid = fig.canvas.mpl_connect('motion_notify_event', on_move_time)

    return vline, text, cid
       
def only_keep_lines(ax,list_lines):
    for line in ax.get_lines():
        if line not in list_lines:
            line.remove()
        
def resetplot(axs,data,TR):
    global line_current,line_brain_ave,brain_ave,t,line_cvr_ref,text_coord
    global line_aif_ave
    global vline1, vline2
    ijk_lines.clear()
    axs.cla()

    h_box = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height
    axs.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axs.xaxis.set_tick_params(labelcolor='white',labelsize=2*h_box,color='none',pad=-15)
    axs.yaxis.set_tick_params(labelcolor='white',labelsize=2*h_box,color='none',pad=-20)
    text_coord = axs.text(0.6, 0.95, '', fontsize=2.5*h_box, color='white', ha='left', va='top', transform=axs.transAxes)
    for spine in axs.spines.values():
        spine.set_visible(False)  # Completely hide the spines
    vline, text, cid = on_move_time_set(axs)

     
    nt = data.shape[3]
    t = TR*np.arange(0,nt)
    line_current, = axs.plot(t,data[I0,J0,K0,:],'w:',marker='.', markerfacecolor='none', markeredgecolor='none',zorder=500)
    brain_ave = np.ma.mean(data,axis=(0,1,2))
    np.savetxt(os.path.join(dir_perfmri,'brain_ave.1D'),np.array(brain_ave).transpose())
    line_brain_ave, = axs.plot(t, brain_ave, color='#00FF00', linestyle='-', linewidth=8, alpha=0.5, zorder=600,marker='.', markerfacecolor='red', markeredgecolor='none')
    
    
    # Re-display data
    # Display data 
    data_mean = np.ma.mean(data,axis=3)
    min_data, max_data = vminvmax_percentile(data_mean,3,3)
    _, _, K = ijk_anat2func([0, 0, slice_n.val]) 
    ove1.set_data(data_mean[:,:,K])
    ove1.set_clim(vmin=min_data, vmax=max_data)


    # Re-plot the aif lines and its average based on aif mask
    ii,jj,kk = np.where(~aif.mask)
    for i, j, k in zip(ii, jj, kk):
        # Plot the time series of data at coordinate (i, j, k)
        key = f"{i}_{j}_{k}"
        l = ax3.plot(t,data[i, j, k, :],picker=True, pickradius=3)
        ijk_lines.append([key, l[0]])  # Append the key and plot object pair to the ijk_lines list
    
    aif_ave = mask_average(data,aif)    
    line_aif_ave, = ax3.plot(t,aif_ave, color='#FF00FF', linestyle='-', linewidth=8, alpha=0.4, zorder=600)
    line_aif_ave.set_visible(chk_ave_aif_state.get())
    
    # Make 2 lines for trim and scaling
    vline1 = ax3.axvline(0, color='green', linestyle='-',zorder=700)
    vline2 = ax3.axvline(10, color='green', linestyle='-',zorder=701)

    # Reference for fMRI
    transform = blended_transform_factory(axs.transData, axs.transAxes)
    line_cvr_ref = Line2D([],[], transform=transform, color='red', lw=1,zorder=800)
    axs.add_line(line_cvr_ref)

    if chk_autoscale_state.get():
        axs.relim()        # Recompute limits based on all data (even those out of view)
        axs.autoscale() 
    plt.draw()
    

def find_bolus_lines(bolus,TR):
    # Find the bolus start and end times
    i_start_bolus, _, i_end_bolus = bolus_times_keys(bolus,TR)
    i_end_base = i_start_bolus - round(3/TR) # widraw 3 sec to be safe as fast arterial signal have early "end of baseline"
    i_end_base = max(i_end_base,0) 
    return i_end_base, i_end_bolus

# This function is to be used on a smooth bolus like the average brain bolus
def bolus_times_keys(bolus,TR):
    i_start_bolus, _ = bolus_times_1d(bolus,TR,10)
    i_start_bolus = round(i_start_bolus/TR)
    i_max_bolus = np.argmax(bolus)
    i_end_bolus = argrelmin(bolus[i_max_bolus:-1],order=2)[0][0] + i_max_bolus
    return i_start_bolus, i_max_bolus, i_end_bolus

def bolus_times_1d(bolus, TR, perc):
    i_max = np.argmax(bolus)
    v_max = bolus[i_max]
    
    # Calculate diff without using masked arrays, use boolean indexing instead
    threshold = v_max * perc / 100
    diff = bolus - threshold
    diff[:i_max + 1][diff[:i_max + 1] < 0] = np.inf  # Set negative differences to inf

    # Find t1 using np.argmin
    ip1 = np.argmin(diff[:i_max + 1])
    if ip1 == 0:
        t1 = 0
    else:
        h = bolus[ip1] - threshold
        if bolus[ip1] != bolus[ip1 - 1]:
            H = bolus[ip1] - bolus[ip1 - 1]
            d = TR * (h / H)  # Using Thales theorem
            t1 = ip1 * TR - d
        else:
            t1 = ip1 * TR

    # Now for t2
    diff[i_max:][diff[i_max:] < 0] = np.inf  # Set negative differences to inf
    ip2 = np.argmin(diff[i_max:]) + i_max

    if ip2 == len(bolus) - 1:
        t2 = ip2 * TR
    else:
        h = bolus[ip2] - threshold
        if bolus[ip2] != bolus[ip2 + 1]:
            H = bolus[ip2] - bolus[ip2 + 1]
            d = TR * (h / H)  # Using Thales theorem
            t2 = ip2 * TR + d
        else:
            t2 = (ip2 + 1) * TR

    return t1, t2

def distance_to_line(xdata, line_x):
    return abs(xdata - line_x)

def on_move_time(event):
    if event.inaxes == ax3 and event.button == 3:  # Right-click is button 3
        mouse_x = event.xdata
        # Update the position of the selected line
        if selected_line == 'vline1':
            vline1.set_xdata([mouse_x, mouse_x])
        elif selected_line == 'vline2':
            vline2.set_xdata([mouse_x, mouse_x])
        plt.draw()

def on_click_time(event):
    global selected_line
    # Check if right-click is pressed and mouse is in the axes
    if event.inaxes == ax3 and event.button == 3:
        mouse_x = event.xdata

        # Determine the closest line to the click position
        dist_vline1 = distance_to_line(mouse_x, vline1.get_xdata()[0])
        dist_vline2 = distance_to_line(mouse_x, vline2.get_xdata()[0])

        # Select the line that is closer
        min_dist = np.min([dist_vline1,dist_vline2])
        if min_dist == dist_vline1:
            selected_line = 'vline1'
        elif min_dist == dist_vline2: 
            selected_line = 'vline2'
        

def on_release_time(event):
    # Release the line (stop moving) when right mouse button is released
    if event.inaxes == ax3 and event.button == 3: 
        selected_line = None




def bolus_times_3d(array_4d, TR, input_mask, perc):
    """
    Vectorized calculation of t1 and t2 times for the bolus curve for each voxel in the 4D array.
    
    Parameters:
    array_4d : np.ndarray
        A 4D array (ni, nj, nk, nt) representing the data.
    mask : np.ndarray
        A 3D array (ni, nj, nk) representing the mask (non-zero values for valid voxels).
    TR : float
        The repetition time (TR).
    perc : float
        The percentage of the maximum bolus value used to define the threshold.
        
    Returns:
    t1_3d : np.ma.MaskedArray
        The t1 times for each voxel.
    t2_3d : np.ma.MaskedArray
        The t2 times for each voxel.
    """
    
    # Get the shape of the 4D array
    ni, nj, nk, nt = array_4d.shape

    # Get the 1/0 input mask
    mask = np.asarray(input_mask)
    
    # Identify the voxels within the mask (non-zero mask values)
    #voxels = np.argwhere(mask != 0)  # List of (i, j, k) indices
    voxels = np.argwhere(mask != 0)
    num_voxels = voxels.shape[0]

    # Reshape the 4D array into a 2D array for processing (voxel, time)
    bolus_curves = array_4d[mask != 0, :]  # Shape: (num_voxels, nt)

    # Find the index of the maximum for each voxel (voxel-wise argmax)
    i_max = np.argmax(bolus_curves, axis=1)  # Shape: (num_voxels,)
    i_max = i_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Find the maximum value for each voxel
    v_max = np.max(bolus_curves, axis=1)
    v_max = v_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Calculate the threshold for each voxel
    threshold = v_max * perc / 100  # Shape: (num_voxels, 1)
    threshold = np.repeat(threshold, nt, axis=1)  # Shape: (num_voxels, nt)    

    # Compute the difference with the threshold
    step = np.where(bolus_curves > threshold, 1, -1) # Shape: (num_voxels, nt)
    diff_step = step[:,1:] - step[:,:-1]
    diff_step_mask = (diff_step == 0) # Shape: (num_voxels, nt-1)
    print("diff_step_mask=",diff_step_mask.shape,"numvox=",num_voxels)
    diff_step_mask = np.concatenate([diff_step_mask,np.ones((num_voxels, 1), dtype=bool)], axis=1)  # Shape: (num_voxels, nt)
   

    # Create masks before and after the max for each voxel (broadcast i_max)
    mask_before_max = np.arange(nt)[np.newaxis, :] < i_max  # Shape: (num_voxels, nt)
    mask_after_max = np.arange(nt)[np.newaxis, :] >= i_max  # Shape: (num_voxels, nt)



    nt_indices = np.tile(np.arange(nt), (num_voxels, 1)) # Shape: (num_voxels, nt)
    

    mask1 = np.logical_or(diff_step_mask, mask_after_max)
    mask2 = np.logical_or(diff_step_mask, mask_before_max)
    
   

    nt_indices_masked_after_max = np.ma.masked_array(nt_indices,mask=mask1) # Shape: (num_voxels, nt)
    nt_indices_masked_before_max = np.ma.masked_array(nt_indices,mask=mask2) # Shape: (num_voxels, nt)

    # Find the closest points to the threshold (argmin on absolute differences)
    ip1 = np.ma.max(nt_indices_masked_after_max, axis=1) # Shape: (num_voxels,)
    ip1 = np.ma.filled(ip1,0)
    ip2 = np.ma.min(nt_indices_masked_before_max, axis=1) # Shape: (num_voxels,)
    ip2 = np.ma.filled(ip2,nt-1)
    
    
    

    
    # Get the index array for voxels
    vi = np.arange(bolus_curves.shape[0])  
    
    ip1_next = np.ma.where(ip1 + 1 < nt - 1, ip1 + 1, nt - 1)
    H1 = np.ma.array(bolus_curves[vi, ip1_next] - bolus_curves[vi, ip1])
    h1 = np.maximum(threshold[vi, 0] - bolus_curves[vi,ip1],0) # bolus_curves of ip1=0 could be above threshold (whereas it should be under) ... in this case, h1 is set to 0
    R1 = np.where(H1 == 0, 0.5, h1 / H1)  # Element-wise operation
    
    ip2_next = np.where(ip2 + 1 < nt - 1, ip2 + 1, nt - 1)
    H2 = np.ma.array(bolus_curves[vi,ip2] - bolus_curves[vi,ip2_next])
    h2 = bolus_curves[vi,ip2] - threshold[vi, 0]
    R2 = np.where(H2 == 0, 0.5, h2 / H2)  # Element-wise operation

    # Apply Thales theorem to compute the fractional time adjustment and add it to ip1/ip2
    t1 =  TR * (ip1 + R1)
    t1[i_max.flatten() == 0] = TR/10000  # voxels where max is first frame, set to ~ 0
    t2 =  TR * (ip2 + R2)
    
    # Reshape t1 and t2 back to 3D
    t1_3d = np.zeros(mask.shape)
    t2_3d = np.zeros(mask.shape)

    # Fill in the voxels for which the mask is not zero
    t1_3d[mask != 0] = t1
    t2_3d[mask != 0] = t2


    return np.ma.masked_array(t1_3d, mask=(mask == 0)), np.ma.masked_array(t2_3d, mask=(mask == 0))


def bolus_times_3d_2(array_4d, TR, mask, perc):
    """
    Vectorized calculation of t1 and t2 times for the bolus curve for each voxel in the 4D array.
    
    Parameters:
    array_4d : np.ndarray
        A 4D array (ni, nj, nk, nt) representing the data.
    mask : np.ndarray
        A 3D array (ni, nj, nk) representing the mask (non-zero values for valid voxels).
    TR : float
        The repetition time (TR).
    perc : float
        The percentage of the maximum bolus value used to define the threshold.
        
    Returns:
    t1_3d : np.ma.MaskedArray
        The t1 times for each voxel.
    t2_3d : np.ma.MaskedArray
        The t2 times for each voxel.
    """
    
    # Get the shape of the 4D array
    ni, nj, nk, nt = array_4d.shape

    # Identify the voxels within the mask (non-zero mask values)
    voxels = np.argwhere(mask != 0)  # List of (i, j, k) indices
    num_voxels = voxels.shape[0]

    # Reshape the 4D array into a 2D array for processing (voxel, time)
    bolus_curves = array_4d[mask != 0, :]  # Shape: (num_voxels, nt)

    # Find the index of the maximum for each voxel (voxel-wise argmax)
    i_max = np.argmax(bolus_curves, axis=1)  # Shape: (num_voxels,)
    i_max = i_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Find the maximum value for each voxel
    v_max = np.max(bolus_curves, axis=1)
    v_max = v_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Calculate the threshold for each voxel
    threshold = v_max * perc / 100  # Shape: (num_voxels, 1)
    threshold = np.repeat(threshold, nt, axis=1)  # Shape: (num_voxels, nt)    

    # Compute the difference with the threshold
    step = np.where(bolus_curves > threshold, 1, -1) # Shape: (num_voxels, nt)
    diff_step = step[:,1:] - step[:,:-1]
    diff_step_mask = (diff_step == 0) # Shape: (num_voxels, nt-1)
    diff_step_mask = np.concatenate([diff_step_mask,np.ones((num_voxels, 1), dtype=bool)], axis=1)  # Shape: (num_voxels, nt)
   

    # Create masks before and after the max for each voxel (broadcast i_max)
    mask_before_max = np.arange(nt)[np.newaxis, :] < i_max  # Shape: (num_voxels, nt)
    mask_after_max = np.arange(nt)[np.newaxis, :] >= i_max  # Shape: (num_voxels, nt)



    nt_indices = np.tile(np.arange(nt), (num_voxels, 1)) # Shape: (num_voxels, nt)
    

    mask1 = np.logical_or(diff_step_mask, mask_after_max)
    mask2 = np.logical_or(diff_step_mask, mask_before_max)
    
   

    nt_indices_masked_after_max = np.ma.masked_array(nt_indices,mask=mask1) # Shape: (num_voxels, nt)
    nt_indices_masked_before_max = np.ma.masked_array(nt_indices,mask=mask2) # Shape: (num_voxels, nt)

    # Find the closest points to the threshold (argmin on absolute differences)
    ip1 = np.ma.max(nt_indices_masked_after_max, axis=1) # Shape: (num_voxels,)
    ip1 = np.ma.filled(ip1,0)
    ip2 = np.ma.min(nt_indices_masked_before_max, axis=1) # Shape: (num_voxels,)
    ip2 = np.ma.filled(ip2,nt-1)
    
    
    # Get the index array for voxels
    vi = np.arange(bolus_curves.shape[0])  
    
    ip1_next = np.ma.where(ip1 + 1 < nt - 1, ip1 + 1, nt - 1)
    H1 = np.ma.array(bolus_curves[vi, ip1_next] - bolus_curves[vi, ip1])
    h1 = threshold[vi, 0] - bolus_curves[vi,ip1]
    R1 = np.where(H1 == 0, 0.5, h1 / H1)  # Element-wise operation
    
    ip2_next = np.where(ip2 + 1 < nt - 1, ip2 + 1, nt - 1)
    H2 = np.ma.array(bolus_curves[vi,ip2] - bolus_curves[vi,ip2_next])
    h2 = bolus_curves[vi,ip2] - threshold[vi, 0]
    R2 = np.where(H2 == 0, 0.5, h2 / H2)  # Element-wise operation

    # Apply Thales theorem to compute the fractional time adjustment and add it to ip1/ip2
    t1 =  TR * (ip1 + R1)
    t1[i_max.flatten() == 0] = 0.0  # voxels where max is first frame]
    t2 =  TR * (ip2 + R2)
    
    # Reshape t1 and t2 back to 3D
    t1_3d = np.zeros(mask.shape)
    t2_3d = np.zeros(mask.shape)

    # Fill in the voxels for which the mask is not zero
    t1_3d[mask != 0] = t1
    t2_3d[mask != 0] = t2
    

    return np.ma.masked_array(t1_3d, mask=(mask == 0)), np.ma.masked_array(t2_3d, mask=(mask == 0))

def bolus_times_3d_old(array_4d, TR, mask, perc):
    """
    Vectorized calculation of t1 and t2 times for the bolus curve for each voxel in the 4D array.
    
    Parameters:
    array_4d : np.ndarray
        A 4D array (ni, nj, nk, nt) representing the data.
    mask : np.ndarray
        A 3D array (ni, nj, nk) representing the mask (non-zero values for valid voxels).
    TR : float
        The repetition time (TR).
    perc : float
        The percentage of the maximum bolus value used to define the threshold.
        
    Returns:
    t1_3d : np.ma.MaskedArray
        The t1 times for each voxel.
    t2_3d : np.ma.MaskedArray
        The t2 times for each voxel.
    """
    
    # Get the shape of the 4D array
    ni, nj, nk, nt = array_4d.shape

    # Identify the voxels within the mask (non-zero mask values)
    voxels = np.argwhere(mask != 0)  # List of (i, j, k) indices
    num_voxels = voxels.shape[0]

    # Reshape the 4D array into a 2D array for processing (voxel, time)
    bolus_curves = array_4d[mask != 0, :]  # Shape: (num_voxels, nt)

    # Find the index of the maximum for each voxel (voxel-wise argmax)
    i_max = np.argmax(bolus_curves, axis=1)  # Shape: (num_voxels,)
    i_max = i_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Find the maximum value for each voxel
    v_max = np.max(bolus_curves, axis=1)
    v_max = v_max[:, np.newaxis]  # Shape: (num_voxels, 1)

    # Calculate the threshold for each voxel
    threshold = v_max * perc / 100  # Shape: (num_voxels, 1)
    threshold = np.repeat(threshold, nt, axis=1)  # Shape: (num_voxels, nt)    

    # Compute the difference with the threshold
    step = np.where(bolus_curves > threshold, 1, -1) # Shape: (num_voxels, nt)
    diff_step = step[:,1:] - step[:,:-1]
    diff_step_mask = (diff_step == 0) # Shape: (num_voxels, nt-1)
    diff_step_mask = np.concatenate([diff_step_mask,np.ones((num_voxels, 1), dtype=bool)], axis=1)  # Shape: (num_voxels, nt)
   

    # Create masks before and after the max for each voxel (broadcast i_max)
    mask_before_max = np.arange(nt)[np.newaxis, :] < i_max  # Shape: (num_voxels, nt)
    mask_after_max = np.arange(nt)[np.newaxis, :] >= i_max  # Shape: (num_voxels, nt)



    nt_indices = np.tile(np.arange(nt), (num_voxels, 1)) # Shape: (num_voxels, nt)
    

    mask1 = np.logical_or(diff_step_mask, mask_after_max)
    mask2 = np.logical_or(diff_step_mask, mask_before_max)
    
   

    nt_indices_masked_after_max = np.ma.masked_array(nt_indices,mask=mask1) # Shape: (num_voxels, nt)
    nt_indices_masked_before_max = np.ma.masked_array(nt_indices,mask=mask2) # Shape: (num_voxels, nt)

    # Find the closest points to the threshold (argmin on absolute differences)
    ip1 = np.ma.max(nt_indices_masked_after_max, axis=1) # Shape: (num_voxels,)
    ip1 = np.ma.filled(ip1,0)
    ip2 = np.ma.min(nt_indices_masked_before_max, axis=1) # Shape: (num_voxels,)
    ip2 = np.ma.filled(ip2,nt-1)
    
    # Get the index array for voxels
    vi = np.arange(bolus_curves.shape[0])  
    
    ip1_next = np.ma.where(ip1 + 1 < nt - 1, ip1 + 1, nt - 1)
    
    H1 = np.ma.array(bolus_curves[vi, ip1_next] - bolus_curves[vi, ip1])
    h1 = threshold[vi, 0] - bolus_curves[vi,ip1]
    R1 = np.where(H1 == 0, 0.5, h1 / H1)  # Element-wise operation
    
    ip2_next = np.where(ip2 + 1 < nt - 1, ip2 + 1, nt - 1)
    H2 = np.ma.array(bolus_curves[vi,ip2] - bolus_curves[vi,ip2_next])
    h2 = bolus_curves[vi,ip2] - threshold[vi, 0]
    R2 = np.where(H2 == 0, 0.5, h2 / H2)  # Element-wise operation

    # Apply Thales theorem to compute the fractional time adjustment and add it to ip1/ip2
    t1 =  TR * (ip1 + R1)
    t2 =  TR * (ip2 + R2)
    
    # Reshape t1 and t2 back to 3D
    t1_3d = np.zeros(mask.shape)
    t2_3d = np.zeros(mask.shape)

    # Fill in the voxels for which the mask is not zero
    t1_3d[mask != 0] = t1
    t2_3d[mask != 0] = t2
    
    return np.ma.masked_array(t1_3d, mask=(mask == 0)), np.ma.masked_array(t2_3d, mask=(mask == 0))


def plot_cross(axx, x, y):
    vline = axx.axvline(x, color='blue', linestyle='-')
    hline = axx.axhline(y, color='blue', linestyle='-')
    cross = [vline, hline]
    return cross

def update_cross(cross, x, y):
    vline, hline = cross
    # Update the data of the lines
    vline.set_xdata([x, x])
    hline.set_ydata([y, y])
        

def average_time_series_4d(A,TR,st,w):
    # Initialize an empty list to store each time series
    series_list = []
    # Compute the length of the time dimension
    n = A.shape[3]
    st = int(st // TR)
    w = int(w // TR)
    # Calculate the number of complete windows we can fit
    while st + w <= n:
        # Slice the array for the given time window
        series_list.append(A[..., st:st+w])
        st += w + 1  # Move to the start of the next window

    # Convert list to a numpy array for easier manipulation
    series_array = np.ma.array(series_list)

    # Calculate the average across the time series
    if len(series_array) == 0:
        return np.array([])  # Return empty array if no series were created
    averaged_series = np.ma.mean(series_array, axis=0)
    return averaged_series


# Functions to calculate the maps
def gamma_var_approx_new(t, bat, ttp, fwhm, cmax):
    b = ((fwhm / 2.335) ** 2) * (1 / ttp)
    a = ttp / b

    Nx, Ny, Nz = ttp.shape
    Nt = t.shape[0]  # Assuming t is a 1D array

    t = t.reshape(1, 1, 1, Nt)
    t = np.broadcast_to(t, (Nx, Ny, Nz, Nt))
    a = np.broadcast_to(a[..., np.newaxis], (Nx, Ny, Nz, Nt))
    b = np.broadcast_to(b[..., np.newaxis], (Nx, Ny, Nz, Nt))
    ttp = np.broadcast_to(ttp[..., np.newaxis], (Nx, Ny, Nz, Nt))
    cmax = np.broadcast_to(cmax[..., np.newaxis], (Nx, Ny, Nz, Nt))
    bat = np.broadcast_to(bat[..., np.newaxis], (Nx, Ny, Nz, Nt))

    # Apply element-wise mask and compute gamma variate
    time_mask = t >= bat
    brain_mask = np.broadcast_to(func_mask[..., np.newaxis],(Nx, Ny, Nz, Nt))
    mask = np.logical_and(time_mask, brain_mask)  # final mask: bool
    gam = np.zeros_like(t)  # shape (Nx, Ny, Nz, Nt)

    # Avoid division by zero or negative powers
    safe_ttp = np.where(ttp == 0, 1e-6, ttp)
    safe_b = np.where(b == 0, 1e-6, b)
    T = t - bat
    gam[mask] = (
        ((T[mask]) ** a[mask]) * np.exp(-T[mask] / safe_b[mask])
    )
    auc = trapz(gam, dx=TR, axis=3)

    return gam, auc

def gamma_var_approx_theoratical(t, bat, ttp, fwhm, cmax):
    ttp = ttp - bat
    b = ((fwhm / 2.335) ** 2) * (1 / ttp) * 0.88
    a = ttp / b * 0.9
    Nx, Ny, Nz = ttp.shape
    Nt = t.shape[0]  # Assuming t is a 1D array

    t = t.reshape(1, 1, 1, Nt)
    t = np.broadcast_to(t, (Nx, Ny, Nz, Nt))
    a = np.broadcast_to(a[..., np.newaxis], (Nx, Ny, Nz, Nt))
    b = np.broadcast_to(b[..., np.newaxis], (Nx, Ny, Nz, Nt))
    ttp = np.broadcast_to(ttp[..., np.newaxis], (Nx, Ny, Nz, Nt))
    cmax = np.broadcast_to(cmax[..., np.newaxis], (Nx, Ny, Nz, Nt))
    bat = np.broadcast_to(bat[..., np.newaxis], (Nx, Ny, Nz, Nt))

    # Apply element-wise mask and compute gamma variate
    time_mask = t >= bat
    brain_mask = np.broadcast_to(func_mask[..., np.newaxis],(Nx, Ny, Nz, Nt))
    mask = np.logical_and(time_mask, brain_mask)  # final mask: bool
    gam = np.zeros_like(t)  # shape (Nx, Ny, Nz, Nt)

    # Avoid division by zero or negative powers
    safe_ttp = np.where(ttp == 0, 1e-6, ttp)
    safe_b = np.where(b == 0, 1e-6, b)
    
    T=t-bat
    gam[mask] = (
        cmax[mask] *
        ((T[mask] / safe_ttp[mask]) ** a[mask]) *
        np.exp(-(T[mask] - ttp[mask]) / safe_b[mask])
    )

    auc = trapz(gam, dx=TR, axis=3)

    return gam, auc


def gamma_var_approx(t, bat, ttp, fwhm, cmax):
    b = ((fwhm / 2.335) ** 2) * (1 / ttp)
    a = ttp / b
    Nx, Ny, Nz = ttp.shape
    Nt = t.shape[0]  # Assuming t is a 1D array

    t = t.reshape(1, 1, 1, Nt)
    t = np.broadcast_to(t, (Nx, Ny, Nz, Nt))
    a = np.broadcast_to(a[..., np.newaxis], (Nx, Ny, Nz, Nt))
    b = np.broadcast_to(b[..., np.newaxis], (Nx, Ny, Nz, Nt))
    ttp = np.broadcast_to(ttp[..., np.newaxis], (Nx, Ny, Nz, Nt))
    cmax = np.broadcast_to(cmax[..., np.newaxis], (Nx, Ny, Nz, Nt))
    bat = np.broadcast_to(bat[..., np.newaxis], (Nx, Ny, Nz, Nt))

    # Apply element-wise mask and compute gamma variate
    time_mask = t >= bat
    brain_mask = np.broadcast_to(func_mask[..., np.newaxis],(Nx, Ny, Nz, Nt))
    mask = np.logical_and(time_mask, brain_mask)  # final mask: bool
    gam = np.zeros_like(t)  # shape (Nx, Ny, Nz, Nt)

    # Avoid division by zero or negative powers
    safe_ttp = np.where(ttp == 0, 1e-6, ttp)
    safe_b = np.where(b == 0, 1e-6, b)
    
    T=t
    gam[mask] = (
        cmax[mask] *
        ((T[mask] / safe_ttp[mask]) ** a[mask]) *
        np.exp(-(T[mask] - ttp[mask]) / safe_b[mask])
    )

    auc = trapz(gam, dx=TR, axis=3)

    return gam, auc


def calc_relperf_gamvar_afni():

    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_relative_gamvar, 'conc.nii.gz')
    opts = read_advanced_options()
    npts_gam = opts['npts_gam']
    calc_map_script = os.path.join(script_directory,'afni_gamvar','fit_gamvar.csh')
    calc_map_process = subprocess.Popen(calc_map_script + ' ' + dir_relative_gamvar + ' ' + npts_gam, shell = True)
    calc_map_process.wait()

def calc_relperf_modfree():
    TR = nb_func.header['pixdim'][4]
    
    
    t1_50, t2_50 = bolus_times_3d(data1, TR, func_mask,50)
    t1_95, t2_95 = bolus_times_3d(data1, TR, func_mask,95)
    bat, _ = bolus_times_3d(data1, TR, func_mask,15)


    # Calculate TTP
    ttp = (t1_95+t2_95)/2
    rttp = ttp - bat

    # Calculate FWHM
    fwhm = t2_50 - t1_50

    # Calculate First momemnt
    Nx, Ny, Nz, Nt = data1.shape
    t = TR * np.arange(Nt)             # shape (Nt,)
    fm = np.sum(data1 * t, axis=3) / np.sum(data1, axis=3)


    # Calculate AUC
    #auc = np.sum(data1,axis=3)
    auc = trapz(data1, dx=TR, axis=3)
    # Calculate cmax : maximum concentration
    cmax = np.max(data1, axis=3)

    # Calculate SMS (smoothness)
    # Scale to max
    cmax_4d = np.repeat(cmax[..., np.newaxis], data1.shape[3], axis=3)
    data1_sc = data1 / cmax_4d

    # Calculate first and second derivative
    data1_d1 = data1_sc[...,1:] - data1_sc[...,:-1]
    data1_d2 = data1_d1[...,1:] - data1_d1[...,:-1]
    sms = np.ma.sum(np.abs(data1_d2),axis=3)
    sms = 100/sms  # This is to take the inverse and to have big number = smooth, same direction as F-score if fitting a gammavar


    # SR and PSR calculation
    v1 = round(vline1.get_xdata()[0]/TR)
    v2 = round(vline2.get_xdata()[0]/TR)
    i1, i2 = sorted((v1, v2))
    i1, i2 = np.clip([i1, i2], 0, Nt-1)

    Ssc = np.ma.exp(-data1)
    Ssc_post = np.ma.mean(Ssc[...,i1:i2+1],axis=3)
    Ssc_min = np.ma.min(Ssc,axis=3)
    SR = 100*(Ssc_post-1) # Equivalent to SR = 100*(Spost-Spre)/Spre
    PSR = 100*(Ssc_post-Ssc_min)/(1-Ssc_min) # Equivalent to PSR = 100**(Spost-Smin)/(Spre-Smin)

    # Calculate K2: the Boxerman leakage corection
    sr_mask_sms_val = np.ma.masked_where((func_mask == 0) | (np.abs(SR) > 5), sms) # Keep voxels with less than 5% diff in pre-post baseline bolus 
    sms_th = np.nanpercentile(sr_mask_sms_val.filled(np.nan),10) # Will be used to reject the 10 percentile most rough curves 
    mask3d = np.ma.masked_where(sr_mask_sms_val < sms_th, np.ones_like(SR))
    conc_ref_ave = mask_average(data1,mask3d)
    conc_ref_ave_sum = np.cumsum(conc_ref_ave)
    np.savetxt(os.path.join(dir_relative_modfree,'conc_ref_ave.1D'),conc_ref_ave)
    np.savetxt(os.path.join(dir_relative_modfree,'conc_ref_ave_sum.1D'),conc_ref_ave_sum)
    # Identify voxels within the mask (non-zero mask values)
    voxels = np.argwhere(func_mask != 0)  # List of (i, j, k) indices
    print(func_mask.shape,"+++++++++++++++++++++++++++++++")
    num_voxels = voxels.shape[0]
    Y = data1[func_mask != 0, :].astype(float).T
    X = np.column_stack((conc_ref_ave,conc_ref_ave_sum))
    X = X / np.linalg.norm(X, axis=0)

    # Perform the linear regression for all voxels
    model = LinearRegression(fit_intercept=False, n_jobs=6)
    model.fit(X,Y)  
    print("bcoef_shape",model.coef_[:, 0].shape) 
    # BCOEF
    K1 = np.ma.masked_all(func_mask.shape)
    K2 = np.ma.masked_all(func_mask.shape)
    i, j, k = voxels.T
    K1[i, j, k] = model.coef_[:, 0]
    K2[i, j, k] = model.coef_[:, 1]
    
    conc_k1 = K1[..., None] * X[:,0]
    conc_k2 = K2[..., None] * X[:,1]
    conc_fit  = conc_k1 + conc_k2
    conc_corr = data1 - conc_k2

    # save2nifti(conc_k1,aff_func_orig,form_code_func_orig,dir_relative_modfree,'conc_k1.nii.gz')
    # save2nifti(conc_k2,aff_func_orig,form_code_func_orig,dir_relative_modfree,'conc_k2.nii.gz')
    save2nifti(conc_fit,aff_func_orig,form_code_func_orig,dir_relative_modfree,'conc_fit.nii.gz')


    # Save files
    save2nifti(ttp,aff_func_orig,form_code_func_orig,dir_relative_modfree,'ttp.nii.gz')
    save2nifti(fm,aff_func_orig,form_code_func_orig,dir_relative_modfree,'fm.nii.gz')
    save2nifti(rttp,aff_func_orig,form_code_func_orig,dir_relative_modfree,'rttp.nii.gz')
    save2nifti(sms,aff_func_orig,form_code_func_orig,dir_relative_modfree,'sms.nii.gz')
    save2nifti(t1_50,aff_func_orig,form_code_func_orig,dir_relative_modfree,'rise.nii.gz')
    save2nifti(t2_50,aff_func_orig,form_code_func_orig,dir_relative_modfree,'decay.nii.gz')
    save2nifti(fwhm,aff_func_orig,form_code_func_orig,dir_relative_modfree,'fwhm.nii.gz')
    save2nifti(bat,aff_func_orig,form_code_func_orig,dir_relative_modfree,'bat.nii.gz')
    save2nifti(auc,aff_func_orig,form_code_func_orig,dir_relative_modfree,'auc.nii.gz')
    save2nifti(cmax,aff_func_orig,form_code_func_orig,dir_relative_modfree,'cmax.nii.gz')
    save2nifti(SR, aff_func_orig,form_code_func_orig, dir_relative_modfree, 'sr.nii.gz')
    save2nifti(PSR, aff_func_orig,form_code_func_orig, dir_relative_modfree, 'psr.nii.gz')
    save2nifti(K2,aff_func_orig,form_code_func_orig,dir_relative_modfree,'k2.nii.gz')
    save2nifti(K1,aff_func_orig,form_code_func_orig,dir_relative_modfree,'k1.nii.gz')
    save2nifti(conc_corr,aff_func_orig,form_code_func_orig,dir_relative_modfree,'conc_k2_corr.nii.gz')


def view_relperf():
    global data4, data5, data6 
    global label4, label5, label6
    global mapval4, mapval5, mapval6
    global ove4, ove5, ove6
    global und4, und5, und6
    global context_menu4, context_menu5, context_menu6

    if relperf_method.get() == 'GamVar-afni':
        dir_relative = dir_relative_gamvar
    elif relperf_method.get() == 'Model Free':
        dir_relative = dir_relative_modfree

    _,auc_array,label4 = load_nifti(dir_relative,'auc.nii.gz')
    data4,ove4,und4,mapval4 = show_map(ax4,auc_array,label4,'hot',5,5,anat) 
    
    _,ttp_array,label5 = load_nifti(dir_relative,'ttp.nii.gz')
    data5,ove5,und5,mapval5 = show_map(ax5,ttp_array,label5,'viridis',5,5,anat)
    
    _,bat_array,label6 = load_nifti(dir_relative,'bat.nii.gz')
    data6,ove6,und6,mapval6 = show_map(ax6,bat_array,label6,'viridis',5,5,anat)
    
    # Calculate the approximation of a gamma variate without doing the fit itself
    # This calculation was placed after viewing auc,ttp,bat to avoid delaying the image display.
    _,fwhm_array,_ = load_nifti(dir_relative,'fwhm.nii.gz')
    _,cmax_array,_ = load_nifti(dir_relative,'cmax.nii.gz')
    gam_approx, auc_approx = gamma_var_approx(t,bat_array,ttp_array,fwhm_array,cmax_array)
    save2nifti(gam_approx,aff_func_orig,form_code_func_orig,dir_relative,'gam_appr.nii.gz')
    save2nifti(auc_approx,aff_func_orig,form_code_func_orig,dir_relative,'auc_appr.nii.gz')

    plt.draw()


def press_calc_relperf():
    if relperf_method.get() == 'GamVar-afni':
        os.makedirs(dir_relative_gamvar, exist_ok=True)
        calc_relperf_gamvar_afni()
    elif relperf_method.get() == 'Model Free':
        os.makedirs(dir_relative_modfree, exist_ok=True)
        calc_relperf_modfree()
    view_relperf()
    
    
def press_switch_data():
    global data1
    if switch_data_method.get() == 'GamVar-afni':
        _, data1, label1 = load_nifti(dir_relative_gamvar,'dsc_gamfit_full.nii.gz')
        data1 = mask_4d_from_3d(func_mask==0,data1)
        resetplot(ax3,data1,TR)
        label1_text.set_text(label1)

    elif switch_data_method.get() == 'Original':
        files = glob(os.path.join(dir_preprocess, "func*scl*"))
        print(files)
        orig = max(files)
        _, data1, label1 = load_nifti(dir_preprocess,orig)
        data1 = mask_4d_from_3d(func_mask==0,data1)
        resetplot(ax3,data1,TR)
        label1_text.set_text(label1)
    plt.draw()

def show_map(axx,map_array,label,colorscale,vmin,vmax,anat_array):
    global ove_roi

    map = np.ma.masked_where(map_array==0,map_array)
    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    axx.clear()

    axx.set_xticks([])
    axx.set_yticks([])
    
    if vmin < 0:
        _ , max_func = vminvmax_percentile(map,vmax,vmax)
        min_func = -max_func
    elif vmax < 0: # For CVR corr/BOLD with colorscale ranging from -val to +val
        min_func , _ = vminvmax_percentile(map,vmin,vmin)
        max_func = -min_func
    else:
        min_func , max_func = vminvmax_percentile(map,vmin,vmax)

    if axx == ax4:
        ove_roi = ax4.imshow(roi_map[:, :, k_func],cmap=cmap_roi,alpha=0.7,extent=fov_func,zorder=3,vmin=0.5,vmax=6.5)
    ove = axx.imshow(map[:, :, k_func],cmap=colorscale,vmin = min_func, vmax = max_func, alpha=0.9,extent=fov_func,zorder=2) 
    und = axx.imshow(anat_array[:, :, slice_n.val],cmap='gray',extent=fov_anat,zorder=1,vmin=min_anat,vmax=max_anat)

    xborder = 0.05*(fov_anat[1]-fov_anat[0])
    axx.set_xlim([fov_anat[0], fov_anat[1] + xborder])

    # Find the appropriate font size
    h_box = axx.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height
    
    # Image label
    axx.text(0.05, 0.95,label,color='white', fontsize=3.5*h_box, ha='left', va='top', transform=axx.transAxes)

    # Metrics value in image
    mapval = axx.text(0.05, 0.05,'--',color='white', fontsize=2.5*h_box, ha='left', va='top', transform=axx.transAxes)

    # Set colorscale
    inset_pos = [0.97, 0.15, 0.03, 0.7]  # [left, bottom, width, height]
    
    # Create the inset axes for the colorbar
    cax = axx.inset_axes(inset_pos)

    cbar = plt.colorbar(ove, cax=cax, orientation='vertical')
    ove.cbar = cbar # store cbar as an attribute to ove

    # Fix the number of tick
    tick_fs = 2.5  * h_box
    cbar.ax.tick_params(labelsize=tick_fs, pad=2)

    apply_cbar_format(cbar)
    
    
    global cross1, cross2, cross3, cross4
    if axx == ax5:
        cross2 = plot_cross(ax5,0,0)
    elif axx == ax6:
        cross3 = plot_cross(ax6,0,0)
    elif axx == ax4:
        cross4 = plot_cross(ax4,0,0)
          
   

    plt.draw()

    return map, ove, und, mapval



def calc_aif():
    Nvox=txt_nvox.get("1.0",END)
    Nvox=np.int64(Nvox.strip())

    if switch_data_method.get() == 'GamVar-afni':
        dir_relative = dir_relative_gamvar
    elif switch_data_method.get() == 'Original':
        dir_relative = dir_relative_modfree
    else:
        print("This method is not available yet")

    _,cmax,_ = load_nifti(dir_relative,'cmax.nii.gz')
    _,auc,_ = load_nifti(dir_relative,'auc.nii.gz')
    _,dbase,_ = load_nifti(dir_relative_modfree,'sr.nii.gz')
    dbase = np.abs(dbase)
    _,sms,_ = load_nifti(dir_relative,'sms.nii.gz')
    _,rise,_ = load_nifti(dir_relative,'rise.nii.gz')
    _,decay,_ = load_nifti(dir_relative,'decay.nii.gz')
    _,bat,_ = load_nifti(dir_relative,'bat.nii.gz')
    _,fwhm,_ = load_nifti(dir_relative,'fwhm.nii.gz')
    _,ttp,_ = load_nifti(dir_relative,'ttp.nii.gz')

    # Read the options and set the variables accordingly.
    opts = read_advanced_options()
    sr_perc = float(opts['sr_perc'])
    sms_perc = float(opts['sms_perc'])
    amp_perc = float(opts['amp_perc'])
    amp = auc if opts['amp'] == "auc" else cmax

    tp1 = locals()[opts['time_para1']]
    tp2 = locals()[opts['time_para2']]
    tp3 = locals()[opts['time_para3']]



    dbase_mask_sms_val = np.ma.masked_where((func_mask == 0) | (np.abs(dbase) > sr_perc), sms)
    sms_th = np.nanpercentile(dbase_mask_sms_val.filled(np.nan),sms_perc)

    sms_mask_amp_val = np.ma.masked_where(dbase_mask_sms_val < sms_th, amp)
    

    amp_th = np.nanpercentile(sms_mask_amp_val.filled(np.nan),100-amp_perc)
    

    amp_mask_tp1_val = np.ma.masked_where(sms_mask_amp_val < amp_th, tp1)
    
    N = np.ma.count(amp_mask_tp1_val)
    
    # Calculate the "shared" percentage in order to obatined Nvox number of AIF.
    perc = 100 * math.exp((1/3) * math.log(Nvox / N))
    

    tp1_th = np.nanpercentile(amp_mask_tp1_val.filled(np.nan),perc)

    tp1_mask_tp2_val = np.ma.masked_where(amp_mask_tp1_val > tp1_th, tp2)
    ave = mask_average(data1,tp1_mask_tp2_val)
    tp2_th = np.nanpercentile(tp1_mask_tp2_val.filled(np.nan),perc)
    tp2_mask_tp3_val = np.ma.masked_where(tp1_mask_tp2_val > tp2_th, tp3)
    ave = mask_average(data1,tp2_mask_tp3_val)

    tp3_th = np.nanpercentile(tp2_mask_tp3_val.filled(np.nan),perc)
    aif[:] = np.ma.masked_where(tp2_mask_tp3_val > tp3_th, np.ones_like(tp3))[:]


    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    ove1b.set_data(aif[:, :, k_func])



     # Here I remove all the lines except for the time-line, the dotted current line and the average brain line.  I remove the average AIF, since it is re-calculated here.  Since I remove the average AIF, I need to add the line : ax3.add_line(line_aif_ave)
    resetplot(ax3,data1,TR)
    
    view_aif()
    

    # Average AIF signal where aif is not masked
    aif_ave = mask_average(data1,aif)
    line_aif_ave.set_ydata(aif_ave)
    line_aif_ave.set_xdata(t)
    ax3.add_line(line_aif_ave) # The average AIF line was removed from the plot, so I add it again.
    line_aif_ave.set_visible(chk_ave_aif_state.get())

    save_aif()
    plt.draw()

# For now, this function is not used
def calc_vof():
    Nvox=txt_nvox.get("1.0",END)
    Nvox=np.int64(Nvox.strip())

    if switch_data_method.get() == 'GamVar-afni':
        dir_relative = dir_relative_gamvar
    elif switch_data_method.get() == 'Original':
        dir_relative = dir_relative_modfree
    else:
        print("This method is not available yet")

    _,cmax,_ = load_nifti(dir_relative,'cmax.nii.gz')
    _,auc,_ = load_nifti(dir_relative,'auc.nii.gz')
    _,dbase,_ = load_nifti(dir_relative_modfree,'sr.nii.gz')
    dbase = np.abs(dbase)
    _,sms,_ = load_nifti(dir_relative,'sms.nii.gz')
    _,rise,_ = load_nifti(dir_relative,'rise.nii.gz')
    _,decay,_ = load_nifti(dir_relative,'decay.nii.gz')
    _,bat,_ = load_nifti(dir_relative,'bat.nii.gz')
    _,fwhm,_ = load_nifti(dir_relative,'fwhm.nii.gz')
    _,ttp,_ = load_nifti(dir_relative,'ttp.nii.gz')

    # Read the options and set the variables accordingly.
    opts = read_advanced_options()
    sr_perc = float(opts['sr_perc'])
    sms_perc = float(opts['sms_perc'])
    amp_perc = float(opts['amp_perc'])
    amp = auc if opts['amp'] == "auc" else cmax

    tp1 = locals()[opts['time_para1']]
    tp2 = locals()[opts['time_para2']]
    tp3 = locals()[opts['time_para3']]



    dbase_mask_sms_val = np.ma.masked_where((func_mask == 0) | (np.abs(dbase) > sr_perc), sms)
    sms_th = np.nanpercentile(dbase_mask_sms_val.filled(np.nan),sms_perc)

    sms_mask_amp_val = np.ma.masked_where(dbase_mask_sms_val < sms_th, amp)
    

    amp_th = np.nanpercentile(sms_mask_amp_val.filled(np.nan),100-amp_perc)
    

    amp_mask_tp1_val = np.ma.masked_where(sms_mask_amp_val < amp_th, tp1)
    
    N = np.ma.count(amp_mask_tp1_val)
    
    # Calculate the "shared" percentage in order to obatined Nvox number of AIF.
    perc = 100 * math.exp((1/3) * math.log(Nvox / N))
    

    tp1_th = np.nanpercentile(amp_mask_tp1_val.filled(np.nan),100-perc)
    tp1_mask_tp2_val = np.ma.masked_where(amp_mask_tp1_val < tp1_th, tp2)
    tp2_th = np.nanpercentile(tp1_mask_tp2_val.filled(np.nan),100-perc)
    tp2_mask_tp3_val = np.ma.masked_where(tp1_mask_tp2_val < tp2_th, tp3)
    tp3_th = np.nanpercentile(tp2_mask_tp3_val.filled(np.nan),100-perc)
    aif[:] = np.ma.masked_where(tp2_mask_tp3_val < tp3_th, np.ones_like(tp3))[:]


    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    ove1b.set_data(aif[:, :, k_func])



     # Here I remove all the lines except for the time-line, the dotted current line and the average brain line.  I remove the average AIF, since it is re-calculated here.  Since I remove the average AIF, I need to add the line : ax3.add_line(line_aif_ave)
    resetplot(ax3,data1,TR)
    
    view_aif()
    

    # Average AIF signal where aif is not masked
    aif_ave = mask_average(data1,aif)
    line_aif_ave.set_ydata(aif_ave)
    line_aif_ave.set_xdata(t)
    ax3.add_line(line_aif_ave) # The average AIF line was removed from the plot, so I add it again.
    line_aif_ave.set_visible(chk_ave_aif_state.get())

    save_aif()
    plt.draw()

def press_calc_aif_cvr():
    opts = read_advanced_options()
    if opts['aif_cvr_method'] == "1":
        calc_aif_cvr_method1()
    elif opts['aif_cvr_method'] == "2":
       calc_aif_cvr_method2() 
    
def calc_aif_cvr_method2():
      
    Nvox=txt2_nvox.get("1.0",END)
    Nvox=np.int64(Nvox.strip())

    _,bold,_ = load_nifti(dir_cvr,'bold.nii.gz')
    bold = np.ma.masked_where(func_mask==0,bold)
    _,corr,_ = load_nifti(dir_cvr,'rcoef.nii.gz')
    corr = np.ma.masked_where(func_mask==0,corr)
    _,lag,_ = load_nifti(dir_cvr,'lag.nii.gz')
    lag = np.ma.masked_where(func_mask==0,lag)

    # Taking the inverse function, so I can use mask_by_percent whether inv_lag is the
    # cvr_p1, cvr_p2 or cvr_p3
    
    rev_lag = np.max(lag)+0.1-lag # So that rev_lag is never 0 avoid to be masked out
    print("REVREVREV",type(rev_lag))
    save2nifti(rev_lag,aff_func_orig,form_code_func_orig, dir_cvr, 'rev_lag.nii.gz')

    if type_aif.get() == "Neg":
        bold = -bold
        corr = -corr

    # smoothing kernel fwhm = 10mm => sigma = 10/2.333
    vox_size = np.mean(nb_func.header['pixdim'][1:4])
    sig = vox_size/2.333
    sm_lag = gaussian_filter(lag,sigma = (sig,sig,sig))
    sm_mask = gaussian_filter(np.ma.filled(func_mask,0),sigma = (sig,sig,sig)) 
    sm_lag = np.ma.masked_where(lag==0,sm_lag / sm_mask)
    rev_sm_lag = np.max(sm_lag)-sm_lag
    
    save2nifti(rev_sm_lag,aff_func_orig,form_code_func_orig,dir_cvr,'rev_sm_lag.nii.gz')
    
    opts = read_advanced_options()
    params = opts['method2_param_order']
    params = params.replace("lag", "rev_sm_lag")
    parts = params.split()
    cvr_p1 = locals()[parts[0]]
    cvr_p2 = locals()[parts[1]]
    cvr_p3 = locals()[parts[2]]
    
    N = np.ma.count(bold)
    N_metrics = 3
    perc = 100 * (Nvox / N) ** (1/N_metrics)

    cvr_p2_masked  = mask_by_percentile(cvr_p1,cvr_p2,100-perc,"low")
    cvr_p3_masked = mask_by_percentile(cvr_p2_masked,cvr_p3,100-perc,"low")
    aif[:] = mask_by_percentile(cvr_p3_masked, np.ones_like(func_mask),100-perc,"low")[:]

    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    ove1b.set_data(aif[:, :, k_func])

    # Here I remove all the lines except for the time-line, the dotted current line and the average brain line.  I remove the average AIF, since it is re-calculated here.  Since I remove the average AIF, I need to add the line : ax3.add_line(line_aif_ave)
    resetplot(ax3,data1,TR)
    #average AIF line was removed from the plot, so I add it again.
    line_aif_ave.set_visible(chk_ave_aif_state.get())
    # Save AIF to file
    save_aif()
    plt.draw()


def calc_aif_cvr_method1():
    
    Nvox=txt2_nvox.get("1.0",END)
    Nvox=np.int64(Nvox.strip())

    _,bold,_ = load_nifti(dir_cvr,'bold.nii.gz')
    bold = np.ma.masked_where(func_mask==0,bold)
    _,corr,_ = load_nifti(dir_cvr,'rcoef.nii.gz')
    corr = np.ma.masked_where(func_mask==0,corr)
    _,lag,_ = load_nifti(dir_cvr,'lag.nii.gz')
    lag = np.ma.masked_where(func_mask==0,lag)
    
    
    if type_aif.get() == "Neg":
        bold = -bold
        corr = -corr

    opts = read_advanced_options()
    lag_thr = float(opts['method1_lag_thr'])
    corr = np.ma.masked_where((func_mask == 0) | (lag >= lag_thr), corr)
    
    N = np.ma.count(corr)
    N_metrics = 2
    perc = 100 * (Nvox / N) ** (1/N_metrics)

    bold_masked  = mask_by_percentile(corr,bold,100-perc,"low")
    aif[:] = mask_by_percentile(bold_masked, np.ones_like(func_mask),100-perc,"low")[:]

    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    ove1b.set_data(aif[:, :, k_func])

    # Here I remove all the lines except for the time-line, the dotted current line and the average brain line.  I remove the average AIF, since it is re-calculated here.  Since I remove the average AIF, I need to add the line : ax3.add_line(line_aif_ave)
    resetplot(ax3,data1,TR)
    #average AIF line was removed from the plot, so I add it again.
    line_aif_ave.set_visible(chk_ave_aif_state.get())
    # Save AIF to file
    save_aif()
    plt.draw()

def mask_by_percentile(mask_image, target_image, percentile, mask_type):
    """
    Apply a mask to a target image based on the percentile threshold 
    calculated from a mask image.

    Parameters:
    - mask_image: array-like
        The image used to calculate the percentile threshold. This can be a 
        regular array or a masked array.
    - target_image: array-like
        The image where the mask will be applied. Values will be masked 
        according to the threshold computed from the mask image.
    - percentile: float
        The percentile value (between 0 and 100) to use for the threshold. 
        For example, a percentile of 50 would mask based on the median value.
    - mask_type: str
        Specifies whether to mask values below or above the percentile threshold.
        Should be either 'low' or 'high'.
        - 'low': Mask values below the calculated percentile threshold.
        - 'high': Mask values above the calculated percentile threshold.

    Returns:
    - masked_target_image: np.ma.MaskedArray
        The target image with the mask applied. Elements are masked based on 
        the threshold calculated from the mask image.
    """
    
    # Check if mask_image is a masked array and compute the percentile threshold
    if isinstance(mask_image, np.ma.MaskedArray):
        # If masked, fill the masked areas with NaN and calculate the threshold
        threshold = np.nanpercentile(mask_image.filled(np.nan), percentile)
    else:
        # For non-masked arrays, calculate the threshold directly
        threshold = np.percentile(mask_image, percentile)
    print("Threshold",threshold)
    # Apply the mask to the target image based on the mask_type ('low' or 'high')
    if mask_type == "low":
        # Mask values in target_image where mask_image is less than or equal to the threshold
        masked_target_image = np.ma.masked_where(mask_image <= threshold, target_image)
    elif mask_type == "high":
        # Mask values in target_image where mask_image is greater than or equal to the threshold
        masked_target_image = np.ma.masked_where(mask_image >= threshold, target_image)
    else:
        # Raise an error if the mask_type is invalid
        raise ValueError("mask_type should be 'low' or 'high'")
    return masked_target_image



 

def save_aif():
    mAIF = []   
    lines = [item[1] for item in ijk_lines]
    for line in lines:
        d = line.get_ydata()
        mAIF.append(d)
    
    current_tab_index = nb_analysis.index("current")

    # Get the text of the currently selected tab
    current_tab_text = nb_analysis.tab(current_tab_index, "text")
    
    if current_tab_text == "CVR Analysis":
        dir = dir_cvr
        if type_aif.get() == "Pos":
            mean_aif = 'mean_AIF_pos.1D'
            multi_aif = 'multi_AIF_pos.1D'
        elif type_aif.get() == "Neg":
            mean_aif = 'mean_AIF_neg.1D'
            multi_aif = 'multi_AIF_neg.1D'

    elif current_tab_text == "DSC Analysis":
        if switch_data_method.get() == 'GamVar-afni':
            dir = dir_relative_gamvar
        elif switch_data_method.get() == 'Original':
            dir = dir_relative_modfree
        mean_aif = 'mean_AIF.1D'
        multi_aif = 'multi_AIF.1D'

    np.savetxt(os.path.join(dir,multi_aif),np.array(mAIF).transpose())
    np.savetxt(os.path.join(dir,mean_aif),np.mean(np.array(mAIF),axis=0).transpose())
    save2nifti(aif,aff_func_orig,form_code_func_orig,dir,'aif.nii.gz')

# Function to average AIF signal
def average_aif(aif):
    aif_expanded = np.expand_dims(aif, axis=-1)
    aif_ave = np.ma.mean(np.ma.masked_array(data1, mask=aif_expanded), axis=(0, 1, 2)) 
    return aif_ave  


def mask_average(data_4d, mask_3d):
    """
    Compute average time series over (i,j,k) using a 3D mask.
    """

    # Ensure data is a masked array (does not modify mask)
    data_4d = np.ma.masked_array(data_4d, copy=False)

    # Build a 3D boolean mask
    if np.ma.isMaskedArray(mask_3d):
        # Get the mask; if scalar, convert to full-false mask
        mask_bool_3d = np.ma.getmaskarray(mask_3d)
    else:
        # Zero = masked, nonzero = unmasked
        mask_bool_3d = (mask_3d == 0)

    # Expand mask to 4D
    mask_4d = mask_bool_3d[..., np.newaxis]
    mask_4d = np.repeat(mask_4d, data_4d.shape[-1], axis=3)

    # Apply mask
    masked_data = np.ma.masked_array(data_4d, mask=mask_4d)

    # Average over space
    return masked_data.mean(axis=(0,1,2))

# Quantitative: fit_aif_conv_exp
def calc_quantperf_exp():
    if relperf_method.get() == 'GamVar-afni':
        dir_relative = dir_relative_gamvar
    elif relperf_method.get() == 'Model Free':
        dir_relative = dir_relative_modfree
    
    os.makedirs(dir_quantitative_expon, exist_ok=True)
    
    min_mtt,max_mtt,delta_mtt = 0 , 8 , 0.1
    aif_ave = mask_average(data1,aif)
    aifconv = exponential_convolution(aif_ave,TR,min_mtt,max_mtt,delta_mtt)
    mtt = multi_regress(data1,aifconv,delta_mtt)
    mtt = mtt + 1 # Add 1s to mtt such that the minimum mtt is 1s instead of zero
    _,auc,_ = load_nifti(dir_relative,'auc.nii.gz')
    _, auc_max = vminvmax_percentile(auc,0.1,0.1)
    opts = read_advanced_options()
    rho = float(opts['rho'])
    Kh = float(opts['Kh'])
    cbv = 100*Kh*rho*auc/auc_max # CBV is in % or mL/100g if 100g ~ 100mL of tissue  - kh = 0.7 - hematocrit concetration difference between tissue and vascular voxels
    # For example a grey matter voxel of 5% means 5mL of blood for 100mL of tissue 5mL/100mL - > 5mL/100g
    cbf = 60*cbv/mtt # The 60 is to convert to mL/100g/min
    
    save2nifti(mtt,aff_func_orig,form_code_func_orig,dir_quantitative_expon,'mtt.nii.gz')
    save2nifti(cbv,aff_func_orig,form_code_func_orig,dir_quantitative_expon,'cbv.nii.gz')
    save2nifti(cbf,aff_func_orig,form_code_func_orig,dir_quantitative_expon,'cbf.nii.gz')
    view_quantperf()
    # Saving to NIFTI
    window.after(0, popup.destroy)
    

    

def calc_quantperf_SVD():

    if relperf_method.get() == 'GamVar-afni':
        dir_relative = dir_relative_gamvar
    elif relperf_method.get() == 'Model Free':
        dir_relative = dir_relative_modfree
    
    os.makedirs(dir_quantitative_decon, exist_ok=True)

    aif_ave = mask_average(data1,aif)
    auc_aif = trapz(aif_ave)
    print("YYYYYYYYYYYYYYYYYYYYYYYVOF", auc_aif)


    _,auc,_ = load_nifti(dir_relative,'auc.nii.gz')
    #auc_max = np.max(auc)
    _, auc_max = vminvmax_percentile(auc,0.1,0.1)
    print("99percentile", auc_max)
    aif_scaled = auc_max*aif_ave / auc_aif
    np.savetxt(os.path.join(dir_quantitative_decon,'scaled_AIF.1D'),np.array(aif_scaled).transpose())


    if quant_method.get() == 'oSVD':
        method = 'osvd'
    elif quant_method.get() == 'SVD':
        method = 'svd'
    
    opts = read_advanced_options()
    svd_threshold = float(opts['svd_threshold'])

    R = deconvolve_brain(data1,aif_scaled,func_mask,method=method,percentage=svd_threshold)
    rho = float(opts['rho'])
    Kh = float(opts['Kh'])
    cbf = np.max(R,axis=3) * 100 * 60 * rho*Kh # mL/g/s *100*60 => mL/100g/min
    mtt = trapz(R, dx=TR, axis=3)/np.max(R,axis=3)
    mtt = np.ma.masked_where(mtt<=0,mtt)
    
    cbv = 100*Kh*rho*auc/auc_max # CBV is in % or mL/100g if 100g ~ 100mL of tissue  - kh = 0.7 - hematocrit concetration difference between tissue and vascular voxels
    # For example a grey matter voxel of 5% means 5mL of blood for 100mL of tissue 5mL/100mL - > 5mL/100g
    
    # Calculate MTT using Central Volume theorem
    mtt_cvt = 60*(cbv/cbf)
    # Calculate CBF using Central Volume theorem
    cbf_cvt = 60*cbv/mtt

    # Calculate Tmax
    t1_95, t2_95 = bolus_times_3d(R, TR, func_mask,95)
    Tmax =(t1_95+t2_95)/2
    
    # Saving to nifti
    save2nifti(R,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'R.nii.gz')
    save2nifti(mtt,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'mtt.nii.gz')
    save2nifti(mtt_cvt,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'mtt_cvt.nii.gz')
    save2nifti(cbf,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'cbf.nii.gz')
    save2nifti(cbv,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'cbv.nii.gz')
    save2nifti(cbf_cvt,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'cbf_cvt.nii.gz')
    save2nifti(Tmax,aff_func_orig,form_code_func_orig,dir_quantitative_decon,'Tmax.nii.gz')
    np.savetxt(os.path.join(dir_quantitative_decon,'scaled_AIF.1D'),np.array(aif_scaled).transpose())
    view_quantperf()
    window.after(0, popup.destroy)

def calc_quantcvr():
    global data1
    os.makedirs(dir_cvr_svd, exist_ok=True)
    _,bold,_ = load_nifti(dir_cvr,'bold.nii.gz')
    # Convert signal to R2*
    sign = np.sign(bold)              # Compute sign of bold (3D)
    #sign = np.repeat(sign[..., np.newaxis],data1.shape[3], axis=3) # Repeat along the new axis to match data shap

    data1_pos = data1 * sign[..., np.newaxis]       # Invert data based on sign
    r2_star = np.ma.log(data1_pos/100 + 1) # Convert to concentration
    
    save2nifti(r2_star,aff_func_orig,form_code_func_orig,dir_cvr_svd,'r2_star0.nii.gz')
    r2_star = np.where(data1_pos >= 0, r2_star, 0)
    save2nifti(r2_star,aff_func_orig,form_code_func_orig,dir_cvr_svd,'r2_star.nii.gz')

    auc = trapz(r2_star, axis=3)
    save2nifti(auc,aff_func_orig,form_code_func_orig,dir_cvr_svd,'auc.nii.gz')
    aif_ave = mask_average(r2_star,aif)
    masked_auc = np.ma.masked_where(aif.mask,auc)
    auc_aif = np.mean(masked_auc)
    _, auc_max = vminvmax_percentile(auc,0.1,0.1)
    #auc_max = np.max(auc)
    aif_scaled = auc_max*aif_ave / auc_aif
    np.savetxt(os.path.join(dir_cvr_svd,'scaled_AIF.1D'),np.array(aif_scaled).transpose())

    if quant_method_cvr.get() == 'oSVD':
        method = 'osvd'
    elif quant_method_cvr.get() == 'SVD':
        method = 'svd'

    opts = read_advanced_options()
    svd_threshold = float(opts['svd_threshold'])
    R = deconvolve_brain(r2_star,aif_scaled,func_mask,method=method,percentage=svd_threshold)
    save2nifti(R,aff_func_orig,form_code_func_orig,dir_cvr_svd,'R.nii.gz')
    rho = float(opts['rho'])
    Kh = float(opts['Kh'])
    cbf = np.max(R,axis=3) * 100 * 60 * Kh*rho # /g/s => /100g/min
    cbf = np.ma.masked_where(bold<=0,cbf) # Mask neg CVR voxels
    save2nifti(cbf,aff_func_orig,form_code_func_orig,dir_cvr_svd,'cbf.nii.gz')
    mtt = TR*np.sum(R,axis=3)/np.max(R,axis=3)
    mtt = np.ma.masked_where(bold<=0,mtt) # Mask neg CVR voxels
    save2nifti(mtt,aff_func_orig,form_code_func_orig,dir_cvr_svd,'mtt.nii.gz')
    cbv = 100*Kh*rho*auc/auc_max # CBV is in % or mL/100g if 100g ~ 100mL of tissue  - kh = 0.7 - hematocrit concetration difference between tissue and vascular voxels
    # For example a grey matter voxel of 5% means 5mL of blood for 100mL of tissue 5mL/100mL - > 5mL/100g
    cbv = np.ma.masked_where(bold<=0,cbv) # Mask neg CVR voxels
    save2nifti(cbv,aff_func_orig,form_code_func_orig,dir_cvr_svd,'cbv.nii.gz')
    cbf_cvt = 60*cbv/mtt
    save2nifti(cbf_cvt,aff_func_orig,form_code_func_orig,dir_cvr_svd,'cbf_cvt.nii.gz')
    # Calculate Tmax
    t1_95, t2_95 = bolus_times_3d(R, TR, func_mask,95)
    Tmax =(t1_95+t2_95)/2
    Tmax = np.ma.masked_where(bold<=0,Tmax) # Mask neg CVR voxels
    save2nifti(Tmax,aff_func_orig,form_code_func_orig,dir_cvr_svd,'Tmax.nii.gz')
    view_quantcvr()

def view_quantcvr():
    global data4  , data5  , data6
    global label4, label5, label6
    global ove4, ove5, ove6
    global und4, und5, und6
    global mapval4, mapval5, mapval6
    global dir_cvr_svd

    _,cbv_array,label4 = load_nifti(dir_cvr_svd,'cbv.nii.gz')
    data4,ove4,und4,mapval4 = show_map(ax4,cbv_array,label4,'hot',5,5,anat)
    
    _,mtt_array,label5 = load_nifti(dir_cvr_svd,'mtt.nii.gz')
    data5,ove5,und5,mapval5 = show_map(ax5,mtt_array,label5,'viridis',5,5,anat)

    _,cbf_array,label6 = load_nifti(dir_cvr_svd,'cbf.nii.gz')
    data6,ove6,und6,mapval6 = show_map(ax6,cbf_array,label6,'hot',5,5,anat)

    plt.draw() 


def press_calc_quantperf():
    if quant_method.get() == 'Residue Exp':
        popup_wait()
        threading.Thread(target=calc_quantperf_exp, daemon=True).start()
    else:
        popup_wait()
        threading.Thread(target=calc_quantperf_SVD, daemon=True).start()

def press_calc_tau_2d():
    global data5,ove5,und5,mapval5
    # Calculate the TAU map
    tau_minmax = box_tau_minmax.get("1.0", "end").strip()  # Strip removes extra newlines
    # Parse the text and assign to variables
    start_text = tau_minmax.split('ST=')[1].split()[0] if 'ST=' in tau_minmax else 0
    start = float(start_text)
    end_text = tau_minmax.split('ET=')[1].split()[0] if 'ET=' in tau_minmax else 8
    end = float(end_text)
    step_text = tau_minmax.split('DT=')[1].split()[0] if 'DT=' in tau_minmax else 0.2
    step = float(step_text)

    # If start is negative, first shift the ref to the left by start
    # And then construct the convolved refs going from 0 to end-start
    if start < 0:
        cvr_ref_sh = multi_shifts_ref(cvr_ref, TR, start,0,10000)
        cvr_ref_sh = cvr_ref_sh.squeeze()
        ref_multi_taus = exponential_convolution(cvr_ref_sh, TR,0,end-start,step)
    else:
        ref_multi_taus = exponential_convolution(cvr_ref, TR, start,end,step)
        print(cvr_ref.shape)

    ni, nj, nk = data1.shape[:3]
    tau = np.ma.masked_all_like(func_mask)
    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    mask_k_func = np.zeros_like(func_mask)
    mask_k_func[:,:,k_func] = func_mask[:,:,k_func]
    _, _, _, _, tau[:,:,:] = cvr_regress_3d(data1,mask_k_func,ref_multi_taus.transpose(),TR)
    tau = tau * step
    data5,ove5,und5,mapval5 = show_map(ax5,tau,"TAU",'viridis',5,5,anat)


def press_calc_tau_3d():
    popup_wait()
    threading.Thread(target=calc_tau_3d, daemon=True).start()

def calc_tau_3d():
    global data5,ove5,und5,mapval5 
    # Calculate the TAU map using exponential_convolution method
    tau_minmax = box_tau_minmax.get("1.0", "end").strip()  # Strip removes extra newlines
    # Parse the text and assign to variables
    start_text = tau_minmax.split('ST=')[1].split()[0] if 'ST=' in tau_minmax else 0
    start = float(start_text)
    end_text = tau_minmax.split('ET=')[1].split()[0] if 'ET=' in tau_minmax else 10
    end = float(end_text)
    step_text = tau_minmax.split('DT=')[1].split()[0] if 'DT=' in tau_minmax else 0.2
    step = float(step_text)
    
    # If start is negative, first shift the ref to the left by start
    # And then construct the convolved refs going from 0 to end-start
    if start < 0:
        cvr_ref_sh = multi_shifts_ref(cvr_ref, TR, start,0,10000)
        cvr_ref_sh = cvr_ref_sh.squeeze()
        ref_multi_taus = exponential_convolution(cvr_ref_sh,TR,0,end-start,step)
    else:
        ref_multi_taus = exponential_convolution(cvr_ref, TR, start,end,step)

    tau = multi_regress(data1, ref_multi_taus, step)
    data5,ove5,und5,mapval5 = show_map(ax5,tau,"TAU",'viridis',5,5,anat)
    window.after(0, popup.destroy)
    save2nifti(tau,aff_func_orig,form_code_func_orig,dir_cvr,'tau.nii.gz')
    np.savetxt(os.path.join(dir_cvr,'ref_convolved.1D'),ref_multi_taus.transpose())
    

def press_view_tau():
    global data5,ove5,und5,mapval5
    _,tau,_ = load_nifti(dir_cvr,'tau.nii.gz')
    data5,ove5,und5,mapval5 = show_map(ax5,tau,"TAU",'viridis',5,5,anat)

def press_calc_lag_2d():
    global data5,ove5,und5,mapval5 
    # Calculate the SHIFT map
    shift_minmax = box_lag_minmax.get("1.0", "end").strip()  # Strip removes extra newlines
    # Parse the text and assign to variables
    start_text = shift_minmax.split('ST=')[1].split()[0] if 'ST=' in shift_minmax else 0
    start = float(start_text)
    end_text = shift_minmax.split('ET=')[1].split()[0] if 'ET=' in shift_minmax else 8
    end = float(end_text)
    step_text = shift_minmax.split('DT=')[1].split()[0] if 'DT=' in shift_minmax else 0.2
    step = float(step_text)
    ref_multi_shifts = multi_shifts_ref(cvr_ref, TR, start, end,step)
    ni, nj, nk = data1.shape[:3]
    lag = np.ma.masked_all_like(func_mask)
    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    mask_k_func = np.zeros_like(func_mask)
    mask_k_func[:,:,k_func] = func_mask[:,:,k_func]
    _, _, _, _, lag[:,:,:] = cvr_regress_3d(data1,mask_k_func,ref_multi_shifts.transpose(),TR)
    lag = lag * step
    data5,ove5,und5,mapval5 = show_map(ax5,lag,"LAG",'viridis',5,5,anat)

def press_calc_lag_3d():
    popup_wait()
    threading.Thread(target=calc_lag_3d, daemon=True).start()

def calc_lag_3d():
    global data4,ove4,und4,mapval4
    global data5,ove5,und5,mapval5
    global data6,ove6,und6,mapval6

    # Calculate the SHIFT map
    shift_minmax = box_lag_minmax.get("1.0", "end").strip()  # Strip removes extra newlines
    # Parse the text and assign to variables
    start_text = shift_minmax.split('ST=')[1].split()[0] if 'ST=' in shift_minmax else 0
    start = float(start_text)
    end_text = shift_minmax.split('ET=')[1].split()[0] if 'ET=' in shift_minmax else 8
    end = float(end_text)
    step_text = shift_minmax.split('DT=')[1].split()[0] if 'DT=' in shift_minmax else 0.2
    step = float(step_text)
    ref_multi_shifts = multi_shifts_ref(cvr_ref, TR, start, end,step)
    np.savetxt(os.path.join(dir_cvr,'ref_multi_shifts.1D'),ref_multi_shifts.transpose())
    #lag = multi_regress(data1,ref_multi_shifts,step)
    bcoef,rcoef,pbold,cnr,ind = cvr_regress_3d(data1,func_mask,ref_multi_shifts.transpose(),TR)
    lag = ind*step
    data4,ove4,und4,mapval4 = show_map(ax4,pbold,"BOLD_LC",fMRI_colors,-9999,5,anat)
    data5,ove5,und5,mapval5 = show_map(ax5,lag,"LAG",'viridis',5,5,anat)
    data6,ove6,und6,mapval6 = show_map(ax6,rcoef,"RCOEF_LC",CoolHot_colors,-9999,0.001,anat)

    window.after(0, popup.destroy)
    
    os.makedirs(dir_cvr_lag,exist_ok=True)
    save2nifti(bcoef,aff_func_orig,form_code_func_orig,dir_cvr_lag,'cvr_lc.nii.gz')
    save2nifti(rcoef,aff_func_orig,form_code_func_orig,dir_cvr_lag,'rcoef_lc.nii.gz')
    save2nifti(pbold,aff_func_orig,form_code_func_orig,dir_cvr_lag,'bold_lc.nii.gz')
    save2nifti(cnr,aff_func_orig,form_code_func_orig,dir_cvr_lag,'cnr_lc.nii.gz')
    save2nifti(lag,aff_func_orig,form_code_func_orig,dir_cvr_lag,'lag.nii.gz')
    
def press_view_lag():
    global data5,ove5,und5,mapval5
    _,lag,_ = load_nifti(dir_cvr,'lag.nii.gz')
    data5,ove5,und5,mapval5 = show_map(ax5,lag,"LAG",'viridis',5,5,anat)


# This function is used to convolve the aif with multiple exponential functions of different mtt (MTT)
def exponential_convolution(aif, TR, mtt_start, mtt_end, dmtt):
    # Define the time array
    L = len(aif)
    t = np.arange(0, L * TR, TR)

    # Define the exponential function characteristics
    mtts = np.arange(mtt_start, mtt_end + dmtt, dmtt)

    # Initialize an empty array to store the results
    aifconvs = np.zeros((len(mtts), 2*len(t)))

    # Pad y L times before with its first elements
    aifpad = np.pad(aif, (L,0), 'constant',constant_values=(aif[0],aif[0]))
    aifconvs[0] = aifpad

    # Loop through each mtt value
    for i, mtt in enumerate(mtts[1:], 1):
        # Define the exponential function
        exponential = np.exp(-t / mtt)
        exponential[t > 5 * mtt] = 0

        # Pad zeros to the exponential function
        exponential = np.pad(exponential, (L,0), 'constant')

        # Normalize the exponential function
        #exponential /= np.sum(exponential)

        # Perform the convolution
        aifconv = convolve(exponential, aifpad, mode='same')

        # Store the result
        aifconvs[i] = aifconv
    
    return aifconvs[:,L:2*L+1]

def multi_shifts_ref(ref, TR, start, end, step):
    # Define the time array
    t = TR*np.arange(0, len(ref))
    shifts = np.arange(start,end,step)
    
    multi_shifts = np.zeros((len(shifts),len(t)))
  
    for i in np.arange(0,len(shifts)):
        f = interp1d(t,ref,kind='linear', fill_value=(ref[0], ref[-1]), bounds_error=False)
        multi_shifts[i,:] = f(t-shifts[i])
    return multi_shifts


def multi_regress(data1, aifconvs, dmtt):
    # Assuming data1 is your masked 4D array and aifconvs is your 2D array
    nx, ny, nz, nt = data1.shape
    nmtt = aifconvs.shape[0]
    Nvox = data1.count()
    # Initialize an array to store the indices of the highest correlation coefficients
    I_AIFCONVS = np.ma.masked_all((nx, ny, nz))
    
    # Iterate over each (x, y, z) position
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not np.ma.is_masked(data1[i, j, k, 0]):
                    # Extract the voxel time series
                    v = data1[i, j, k, :]
                    
                    # Repeat the voxel time series to match the shape of aifconvs
                    v_repeat = np.tile(v, (nmtt, 1))
                    
                    # Compute the correlation coefficients between the voxel time series and each AIF time series
                    corr_matrix = np.corrcoef(v_repeat, aifconvs)
                    corr_coefs = corr_matrix[:nmtt, nmtt:].diagonal()
                    # Find the index of the maximum correlation coefficient
                    best_idx = np.argmax(abs(corr_coefs))
                    
                    # Store the index of the highest correlation coefficient
                    I_AIFCONVS[i, j, k] = best_idx
                    
                    

    # Calculate mtt. Add 0.01 so that AFNI does not interpret those zeros as masked values
    mtt = dmtt * I_AIFCONVS + 0.01
    return mtt



## Deconvoltion

# --- Matrix creation functions ---

def create_block_circulant_matrix(aif):
    """Create a block-circulant matrix (for oSVD)."""
    H = circulant(aif)    
    np.savetxt(os.path.join(dir_quantitative_decon,'H_circulant.1D'),np.array(H))    
    return H

def create_toeplitz_matrix(aif):
    """Create a standard convolution (Toeplitz) matrix (for SVD)."""
    H = toeplitz(aif, np.zeros(len(aif)))
    np.savetxt(os.path.join(dir_quantitative_decon,'H_standard.1D'),np.array(H))    
    return H

# --- Precompute inverse H using SVD ---

def prepare_inverse(H, percentage=20):
    """Compute regularized pseudo-inverse of matrix H using SVD."""
    U, s, Vh = npl.svd(H, full_matrices=False)
    tol = (percentage/100) * np.max(s)
    s_inv = np.array([1/x if x > tol else 0 for x in s])
    return Vh.T @ np.diag(s_inv) @ U.T

# --- Voxel-wise deconvolution ---

def deconvolve_brain(array_4d, aif, mask, method='svd', percentage=20):
    """
    Perform voxel-wise deconvolution on a 4D fMRI array using either standard SVD or oSVD.

    Parameters:
    - array_4d: 4D array of signal (x, y, z, time)
    - aif: 1D arterial input function
    - mask: 3D brain mask
    - method: 'svd' (Toeplitz) or 'osvd' (block circulant)
    - percentage: regularization threshold for singular values

    Returns:
    - R: 4D array of residue functions (same shape as array_4d)
    """
    # Pad aif and array_4d with zeros
    nt = len(aif)
    aif_pad = np.concatenate([np.zeros(nt),aif,np.zeros(nt)])
    array_4d_pad = np.concatenate([np.zeros_like(array_4d),array_4d,np.zeros_like(array_4d)],axis=3)


    if method == 'osvd':
        H = create_block_circulant_matrix(aif_pad)
        print("Block circulant SVD")
    elif method == 'svd':
        H = create_toeplitz_matrix(aif_pad)
        print("Regular SVD")
    else:
        raise ValueError("method must be 'osvd' or 'svd'")

    H_inv = prepare_inverse(H, percentage)

    # Initialize output
    R = np.zeros_like(array_4d_pad, dtype=float)
    ni, nj, nk, _ = R.shape

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if mask[i, j, k]:
                    Ct = array_4d_pad[i, j, k, :]
                    R[i, j, k, :] = H_inv @ Ct
    return R[...,:nt]
    



def view_quantperf():
    global data4  , data5  , data6
    global label4, label5, label6
    global ove4, ove5, ove6
    global und4, und5, und6
    global mapval4, mapval5, mapval6
    global dir_quantitative

    
    if quant_method.get() == 'Residue Exp':
        dir_quantitative = dir_quantitative_expon
    else:
        dir_quantitative = dir_quantitative_decon
    
    _,cbv_array,label4 = load_nifti(dir_quantitative,'cbv.nii.gz')
    data4,ove4,und4,mapval4 = show_map(ax4,cbv_array,label4,'hot',5,5,anat)
    
    _,mtt_array,label5 = load_nifti(dir_quantitative,'mtt.nii.gz')
    data5,ove5,und5,mapval5 = show_map(ax5,mtt_array,label5,'viridis',5,5,anat)

    _,cbf_array,label6 = load_nifti(dir_quantitative,'cbf.nii.gz')
    data6,ove6,und6,mapval6 = show_map(ax6,cbf_array,label6,'hot',5,5,anat)

    plt.draw() 

# CVR regression
def calc_cvr_ref(*args):
    global cvr_ref, cvr_ref_orig, t, shift
    shift = 0

    if type_cvr_ref.get() == "Enter ref file":
        cvr_ref_file = filedialog.askopenfilename(initialdir=maindir,title="Select ref file...")
        cvr_ref = np.loadtxt(cvr_ref_file)
        window.focus_force()

        if len(cvr_ref.shape) == 2:
            t_ref = cvr_ref[:,0] - cvr_ref[0,0]
            cvr_ref = cvr_ref[:,1]

        elif len(cvr_ref.shape) == 1:
            t_ref = TR*np.arange(0,len(cvr_ref))
            
        f = interp1d(t_ref,cvr_ref,kind='linear', fill_value=(cvr_ref[0], cvr_ref[-1]), bounds_error=False)
        cvr_ref = f(t)  
        

    elif type_cvr_ref.get() == "Enter OFF/ON":
        # Ask the user to input a list of numbers as a comma-separated string
        input_str = simpledialog.askstring("Input", "Enter OFF/ON stimulus in seconds (comma-separated):")
        if input_str:
            # Split the input string by commas and convert each part to a float
            try:
                stim_list = [float(num.strip()) for num in input_str.split(',')]
                # Step 1: Generate time array t
                t_fine = np.arange(0, sum(stim_list), 0.1)

                # Step 2: Create cvr_ref_step array
                cvr_ref_step = np.zeros_like(t_fine)
                current_time = 0

                for i, duration in enumerate(stim_list):
                    if i % 2 == 1:  # Set cvr_ref_step to 1 during "on" periods
                        cvr_ref_step[(t_fine >= current_time) & (t_fine < current_time + duration)] = 1
                    current_time += duration

                # Step 3: Convolve with the gamma function
                gamma_function = lambda x: (x**8.6) * np.exp(-x / 0.547)
                cvr_gam = gamma_function(t_fine)
                cvr_conv = convolve(cvr_ref_step, cvr_gam, mode='full')[:len(t_fine)]  # Truncate convolution result
                cvr_conv = cvr_conv/max(cvr_conv)

                # Step 4: Interpolate to match time resolution TR
                interp_func = interp1d(t_fine, cvr_conv, kind='linear',fill_value=(cvr_conv[0], cvr_conv[-1]), bounds_error=False)
                cvr_ref = interp_func(t)

            except ValueError:
                print("Invalid input. Please enter a valid list of numbers.")

    # Save the reference to 1D file for later use
    os.makedirs(dir_cvr,exist_ok=True)
    np.savetxt(os.path.join(dir_cvr,'ref.1D'),cvr_ref)

    # Plot the reference relative to the space of the window.
    cvr_ref_scl = scale_ref(cvr_ref)
    line_cvr_ref.set_data(t,cvr_ref_scl)
    # Make a version cvr_ref_scl for later shift
    cvr_ref_orig = cvr_ref.copy()

    press_calc_regression_2d()          
    plt.draw()

def press_shift_cvr_ref_right():
    global cvr_ref, cvr_ref_orig, shift
    shift += 1
    f = interp1d(t,cvr_ref_orig,kind='linear', fill_value=(cvr_ref_orig[0], cvr_ref_orig[-1]), bounds_error=False)
    cvr_ref = f(t-shift)
    cvr_ref_scl = scale_ref(cvr_ref)
    line_cvr_ref.set_data(t,cvr_ref_scl)
    press_calc_regression_2d()
    plt.draw()

def press_shift_cvr_ref_left():
    global cvr_ref, cvr_ref_orig, shift
    shift -= 1
    f = interp1d(t,cvr_ref_orig,kind='linear', fill_value=(cvr_ref_orig[0], cvr_ref_orig[-1]), bounds_error=False)
    cvr_ref = f(t-shift)
    cvr_ref_scl = scale_ref(cvr_ref)
    line_cvr_ref.set_data(t,cvr_ref_scl)
    press_calc_regression_2d()
    plt.draw()

def scale_ref(ref):
    refmin, refmax = min(ref) , max(ref)
    ref_scl = 0.6*((ref-refmin)/(refmax-refmin)) + 0.2
    return ref_scl

def flip_ref():
    cvr_ref[:] = max(cvr_ref) - cvr_ref
    cvr_ref_scl = scale_ref(cvr_ref)
    if chk_flipref_state.get():
        line_cvr_ref.set_data(t,cvr_ref_scl)
    else:
        line_cvr_ref.set_data(t,cvr_ref_scl)
    plt.draw()

def press_calc_cvr_ref():
    calc_cvr_ref()


# Linear regression analysis
def cvr_regress_3d(array_4d, mask, ref_multi, TR, motparam=None,n_jobs=6):
    # Ensure ref_multi is 2D
    if ref_multi.ndim == 1:
        ref_multi = ref_multi[:, np.newaxis]  # Convert to 2D with shape (nt, 1)
    
    # Get the dimensions of the 4D array
    ni, nj, nk, nt = array_4d.shape
    n_ref = ref_multi.shape[-1]  # This works for both 1D and 2D ref_multi
    
    # Time vector based on TR used for linear drift
    t = TR * np.arange(0, nt)
    
    # Identify voxels within the mask (non-zero mask values)
    voxels = np.argwhere(mask != 0)  # List of (i, j, k) indices
    num_voxels = voxels.shape[0]

    # Reshape the 4D array into a 2D array for regression
    Y = array_4d[mask != 0, :]  # Shape: (num_voxels, nt)
    Y = np.asarray(Y, dtype=float)
    Y = Y.T

    # Prepare arrays to store results for all references
    bcoef_all_refs = np.zeros((num_voxels, n_ref))  # To store coefficients for each reference
    rcoef_all_refs = np.zeros((num_voxels, n_ref))  # To store correlation for each reference
    pbold_all_refs = np.zeros((num_voxels, n_ref))  # To store coefficients for each reference
    cnr_all_refs = np.zeros((num_voxels, n_ref)) 

    # Construct the baseline matrix composed of: the baseline signal (ones), the polynomials (drifts) and the motions
    # See AFNI useful 3dfim+ manual: https://afni.nimh.nih.gov/afni/doc/manual/3dfim+.pdf
    # Construct the drift colums
    tc = t - t.mean() # centered
    ts = tc/np.std(tc) # scaled
    opts = read_advanced_options()
    pol_deg = int(opts['pol_deg'])
    Xdrift = np.column_stack([tc**k for k in range(0, pol_deg + 1)])
    print("Xdrift", Xdrift.shape)
    
    if motparam is not None:
        Xbase = np.column_stack((Xdrift,motparam))
    else:
        Xbase = Xdrift
    print("Xbase", Xbase.shape)
    # Perform linear regression for all voxels
    for r in range(n_ref):
        
        X_r = np.column_stack((ref_multi[:, r], Xbase))
    
        # Perform the linear regression for all voxels
        model = LinearRegression(fit_intercept=False, n_jobs=n_jobs)
        model.fit(X_r, Y)  
        
        # BCOEF
        bcoef_r = model.coef_[:, 0]  # Extract coefficients for the first predictor (ref)
             
        # PBOLD
        min_ref = np.percentile(ref_multi[:, r], 2)
        max_ref = np.percentile(ref_multi[:, r], 98)
        delta_ref = max_ref - min_ref
        pbold_r = bcoef_r * delta_ref

        # CNR
        Y_pred = model.predict(X_r)
        std_residual = np.std((Y - Y_pred), axis=0)
        cnr_r = np.abs(pbold_r) / std_residual 

        # Partial correlation
        # Solve Y ≈ Xbase @ beta_base
        beta_base = np.linalg.lstsq(Xbase, Y, rcond=None)[0]   # (nb, nvox)
        Y_base = Xbase @ beta_base                             # (nt, nvox)
        Y_res = Y - Y_base                                    # (nt, nvox)

        beta_ref_base = np.linalg.lstsq(Xbase, ref_multi[:, r], rcond=None)[0]  # (nb,)
        ref_base = Xbase @ beta_ref_base                                       # (nt,)
        ref_res = ref_multi[:, r] - ref_base                                   # (nt,)

        # Center residuals
        ref_res_c = ref_res - ref_res.mean()
        Y_res_c = Y_res - Y_res.mean(axis=0)

        # Numerator
        num = np.sum(ref_res_c[:, None] * Y_res_c, axis=0)

        # Denominator
        den = np.sqrt(
            np.sum(ref_res_c**2) *
            np.sum(Y_res_c**2, axis=0)
        )

        # Partial correlation
        r_partial = num / den        # shape: (nvox,)

        # Store the results for this reference
        bcoef_all_refs[:, r] = bcoef_r
        rcoef_all_refs[:, r] = r_partial
        pbold_all_refs[:, r] = pbold_r
        cnr_all_refs[:, r] = cnr_r

    # Choose the best reference (highest absolute correlation) for each voxel
    best_ref_idx = np.argmax(np.abs(rcoef_all_refs), axis=1)  # Index of the best ref for each voxel

    # Extract the corresponding best coefficients and correlations
    bcoef_best = bcoef_all_refs[np.arange(num_voxels), best_ref_idx]
    rcoef_best = rcoef_all_refs[np.arange(num_voxels), best_ref_idx]
    pbold_best = pbold_all_refs[np.arange(num_voxels), best_ref_idx]
    cnr_best =   cnr_all_refs[np.arange(num_voxels), best_ref_idx]

    # Initialize the output arrays
    BCOEF = np.ma.masked_all(mask.shape[:3])
    RCOEF = np.ma.masked_all(mask.shape[:3])
    PBOLD = np.ma.masked_all(mask.shape[:3])
    CNR = np.ma.masked_all(mask.shape[:3])
    BEST_I = np.ma.masked_all(mask.shape[:3])

    # Assign the calculated values back to the 3D arrays
    for idx, (i, j, k) in enumerate(voxels):
        if idx < len(bcoef_best):  # Ensure we don't index out of bounds
            BCOEF[i, j, k] = bcoef_best[idx]
            RCOEF[i, j, k] = rcoef_best[idx]
            PBOLD[i, j, k] = pbold_best[idx]
            CNR[i, j, k] = cnr_best[idx]
            BEST_I[i, j, k] = best_ref_idx[idx] + 0.05 # Add 0.05 so it is not 0 that is often (in AFNI) used as mask

    return BCOEF, RCOEF, PBOLD, CNR, BEST_I

def press_calc_regression_2d():
    global data4, data5, data6
    global ove4, ove5, ove6
    global und4, und5, und6
    global label4,label5,label6
    global mapval4, mapval5, mapval6

    motparam = None  # default

    opts = read_advanced_options()
    use_mot = opts['use_motparam']
    motfile = os.path.join(dir_preprocess, 'param_mot.1D')

    if use_mot == 'yes' and os.path.isfile(motfile):
        motparam = np.loadtxt(motfile)


    # Create arrays to hold the coefficients with the same shape as the 3D array
    ni, nj, nk = data1.shape[:3]
    
    BCOEF = np.ma.masked_all_like(func_mask)
    RCOEF = np.ma.masked_all_like(func_mask)
    PBOLD = np.ma.masked_all_like(func_mask)
    CNR =   np.ma.masked_all_like(func_mask)

    _,_,k_func = ijk_anat2func([0,0,slice_n.val])
    mask_k_func = np.zeros_like(func_mask)
    mask_k_func[:,:,k_func] = func_mask[:,:,k_func]

    BCOEF[:,:,:], RCOEF[:,:,:], PBOLD[:,:,:], CNR[:,:,:], _ = cvr_regress_3d(data1,mask_k_func,cvr_ref,TR,motparam=motparam)
    
    data4 ,ove4,und4,mapval4 = show_map(ax4,PBOLD,"BOLD[%]",fMRI_colors,-9999,5,anat)
    data5,ove5,und5,mapval5 = show_map(ax5,CNR,"CNR","inferno",5,5,anat)
    data6,ove6,und6,mapval6 = show_map(ax6,RCOEF,"RCOEF",CoolHot_colors,-9999,0.001,anat)   

def press_calc_regression():
    global data4, data5,data6
    global ove4, ove5, ove6
    global und4, und5, und6
    global label4,label5,label6
    global mapval4, mapval5, mapval6

    motparam = None  # default

    opts = read_advanced_options()
    use_mot = opts['use_motparam']
    motfile = os.path.join(dir_preprocess, 'param_mot.1D')

    if use_mot == 'yes' and os.path.isfile(motfile):
        motparam = np.loadtxt(motfile)

    BCOEF, RCOEF, PBOLD, CNR, _ = cvr_regress_3d(
        data1, func_mask, cvr_ref, TR, motparam=motparam
    )
    # In addition calculate the first moment
    a = data1 - np.min(data1,axis=3,keepdims=True)
    Nt = data1.shape[3]
    t = TR * np.arange(Nt)             # shape (Nt,)
    fm = np.sum(a * t, axis=3) / np.sum(a, axis=3)


    os.makedirs(dir_cvr, exist_ok=True)
    save2nifti(BCOEF,aff_func_orig,form_code_func_orig,dir_cvr,'cvr.nii.gz')
    save2nifti(RCOEF,aff_func_orig,form_code_func_orig,dir_cvr,'rcoef.nii.gz')
    save2nifti(PBOLD,aff_func_orig,form_code_func_orig,dir_cvr,'bold.nii.gz')
    save2nifti(CNR,aff_func_orig,form_code_func_orig,dir_cvr,'cnr.nii.gz')
    save2nifti(fm,aff_func_orig,form_code_func_orig,dir_cvr,'fm.nii.gz')
    np.savetxt(os.path.join(dir_cvr,'ref_shifted.1D'),cvr_ref)
    data4 ,ove4,und4,mapval4 = show_map(ax4,PBOLD,"BOLD[%]",fMRI_colors,-9999,5,anat)
    data5,ove5,und5,mapval5 = show_map(ax5,CNR,"CNR","inferno",5,5,anat)
    data6,ove6,und6,mapval6 = show_map(ax6,RCOEF,"R",CoolHot_colors,-9999,0.001,anat) 

def press_view_regression():
    global data4, data5,data6
    global ove4, ove5, ove6
    global und4, und5, und6
    global label4,label5,label6
    global mapval4, mapval5, mapval6

    _,PBOLD,_ = load_nifti(dir_cvr,'bold.nii.gz')
    _,CNR,_ = load_nifti(dir_cvr,'cnr.nii.gz')
    _,RCOEF,_ = load_nifti(dir_cvr,'rcoef.nii.gz')
    data4 ,ove4,und4,mapval4 = show_map(ax4,PBOLD,"BOLD[%]",fMRI_colors,-9999,5,anat)
    data5,ove5,und5,mapval5 = show_map(ax5,CNR,"CNR","inferno",5,5,anat)
    data6,ove6,und6,mapval6 = show_map(ax6,RCOEF,"R",fMRI_colors,-9999,0.1,anat) 

def press_define_windows(*args):
    global bands

    # --- 1. Safely clear existing bands ---
    if 'bands' in globals() and bands:
        for band in bands:
            try:
                band.remove()
            except Exception:
                pass
        bands.clear()
    else:
        bands = []

    # --- 2. INPUTS mode: user provides duration and start times ---
    if type_bands.get() == "Inputs":
        input_str = simpledialog.askstring(
            "Input",
            "\nEnter duration and start times (sec)\n\nFormat: Duration, t1, t2, t3, ..."
        )

        if input_str:
            try:
                parts = [float(num.strip()) for num in input_str.split(',')]
                if len(parts) < 2:
                    raise ValueError("Need at least one start time after duration.")
                window_width = parts[0]
                start_times = parts[1:]

                for start_time in start_times:
                    band = ax3.axvspan(
                        start_time,
                        start_time + window_width,
                        color='pink', ec='pink', linewidth=3, alpha=0.3
                    )
                    bands.append(band)

            except ValueError as e:
                print("Invalid input:", e)

    # --- 3. AUTO mode: detect peaks automatically ---
    elif type_bands.get() == "Auto":
        input_str = simpledialog.askstring("Input", "\nEnter approximate bolus width (sec)\n")
        if input_str:
            try:
                bolus_width = float(input_str)
                ipeaks, _ = find_peaks(cvr_ref, distance=bolus_width / TR, height=(0.5, None))
                start_times = TR * ipeaks

                for start_time in start_times:
                    band = ax3.axvspan(
                        start_time,
                        start_time + bolus_width,
                        color='pink', ec='pink', linewidth=3, alpha=0.3
                    )
                    bands.append(band)

            except Exception as e:
                print("Error in Auto mode:", e)

    # --- 4. Debug info ---
    if bands:
        print(f"{len(bands)} bands created")
        for i, b in enumerate(bands[:3]):  # show first few only
            print(f"  band {i+1}: {b}")
    else:
        print("No bands created")

    # --- 5. Update plot ---
    ax3.grid(True)
    plt.draw()


def move_bands_right():
    for band in bands:
        band.set_x(band.get_x() + 1)   # Move the band to the right by 1seconds
        plt.draw()

def move_bands_left():
    for band in bands:
        band.set_x(band.get_x() - 1)   # Move the band to the left by 1 seconds
        plt.draw()

def shrink_bands():
    for band in bands:
        band.set_width(band.get_width() - 1)   # Shrink the band by 1 seconds
        plt.draw()

def dilate_bands():
    for band in bands:
        band.set_width(band.get_width() + 1)   # Dilate the band by 1 seconds
        plt.draw()
def press_ave_boluses():
    popup_wait()
    threading.Thread(target=ave_boluses, daemon=True).start()



def ave_boluses():
    global data1, cvr_ref, cvr_ref_orig, pr
    pr += 1

    bd = int(bands[0].get_width() // TR)
    data1_sum = np.zeros_like(data1[...,0:bd]) 
    cvr_ref_sum = np.zeros(bd)
    N_CNR = np.zeros_like(data1[...,0:len(bands)])
    
    i = 0
    for band in bands:
        x = int(band.get_x() // TR)  # Ensure integer division for indexing
        data1_sum = data1_sum + data1[..., x:x+bd]  # Sum over the time window
        if 'cvr_ref' in globals():
            cvr_ref_sum = cvr_ref_sum + cvr_ref[x:x+bd]   
            _,_,_,N_CNR[...,i],_ = cvr_regress_3d(data1_sum/(i+1), func_mask, cvr_ref_sum/(i+1), TR, n_jobs=4)
        i += 1
    
    
    data1 = data1_sum / len(bands)  # Average over the number of bands
    resetplot(ax3,data1,TR)
    window.after(0, popup.destroy)
    save2nifti(data1, aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_ave.nii.gz')
    
    if 'cvr_ref' in globals(): 
        cvr_ref = cvr_ref_sum / len(bands)
        cvr_ref_orig = cvr_ref
        cvr_ref_scl = scale_ref(cvr_ref)
        line_cvr_ref.set_data(t,cvr_ref_scl)
        save2nifti(N_CNR,aff_func_orig,form_code_func_orig,dir_cvr,'cnr_boluses.nii.gz')
    
    

# Slice navigation ===================================================================================
# ====================================================================================================

def update_slice(k_anat):
    for und in [und1, und4, und5, und6]:
        und.set_data(anat[:, :, k_anat])
        
    _, _, k_func = ijk_anat2func([0, 0, k_anat])
    
    if chk_map_state.get():
        visible = 0 <= k_func < data1.shape[2]
    else:
        visible = False

    if chk_func_state.get():
        visible_func = 0 <= k_func < data1.shape[2]
    else:
        visible_func = False
    

    if 'ove1' in globals(): ove1.set_visible(visible_func)
    if 'ove1b' in globals(): ove1b.set_visible(visible)
    if 'ove4' in globals(): ove4.set_visible(visible)
    if 'ove5' in globals(): ove5.set_visible(visible)
    if 'ove6' in globals(): ove6.set_visible(visible)
    #if 'ove_roi' in globals(): ove_roi.set_visible(visible)

    if visible_func:
        l = min(int(vline.get_xdata()[0] // TR), data1.shape[3]-1)
        ove1.set_data(data1[:, :, k_func, l])
    if visible:
        if 'ove4' in globals(): ove4.set_data(data4[:, :, k_func])
        if 'ove5' in globals(): ove5.set_data(data5[:, :, k_func])
        if 'ove6' in globals(): ove6.set_data(data6[:, :, k_func])
        ove1b.set_data(aif[:, :, k_func])
        text_coord.set_text(f'I- J- K{k_func}')
    else:
        text_coord.set_text('I- J- K-')

    # Added for ROI
    visible_roi = 0 <= k_func < data1.shape[2]
    ove_roi.set_visible(visible_roi)
    ove_roi.set_data(roi_map[:, :, k_func])

    fig.canvas.draw_idle()


# Coordinates Conversion functions
# ============================================================================================

def ijk_anat2func(ijk_anat):
    anat2func_vox = npl.inv(aff_func).dot(aff_anat)
    ijk_anat.append(1)
    ijk_func = anat2func_vox@np.array(ijk_anat)
    ijk_func = (np.round(np.array(ijk_func[0:3]))).astype(int)
    ijk_func = ijk_func.tolist()
    return ijk_func

def ijk_func2anat(ijk_func):
    func2anat_vox = npl.inv(aff_anat).dot(aff_func)
    ijk_func.append(1)
    ijk_anat = func2anat_vox@np.array(ijk_func)
    ijk_anat = (np.round(np.array(ijk_anat[0:3]))).astype(int)
    ijk_anat = ijk_anat.tolist()
    return ijk_anat

def xyz2anat_ijk(xyz):
    invaff_anat = npl.inv(aff_anat)
    xyz.append(1)
    ijk_anat = invaff_anat@np.array(xyz)
    ijk_anat = (np.round(np.array(ijk_anat[0:3]))).astype(int)
    ijk_anat = ijk_anat.tolist()
    return ijk_anat

def xyz2func_ijk(xyz):
    invaff_func = npl.inv(aff_func)
    xyz.append(1)
    ijk_func = invaff_func@np.array(xyz)
    ijk_func = (np.round(np.array(ijk_func[0:3]))).astype(int)
    ijk_func = ijk_func.tolist()
    return ijk_func

def xyz2func_ijk_float(xyz):
    invaff_func = npl.inv(aff_func)
    xyz.append(1)
    ijk_func = invaff_func@np.array(xyz)
    ijk_func = np.array(ijk_func[0:3])
    ijk_func = ijk_func.tolist()
    return ijk_func


# Show time series while images navigation ===========================================================
# Important to know the coordoninates of the point (i,j) on the image correspond to
# the i-ieme column and j-ieme line, which is the M(j,i) point in the matrice
# ====================================================================================================

#Show time series while images navigation
def on_mouse_move(event):
    if 'line_current' in globals():
        if event.inaxes in (ax1, ax5, ax6, ax4):
            _, _, K = ijk_anat2func([0, 0, slice_n.val])
            J, I, _ = xyz2func_ijk([-event.xdata, -event.ydata, 0])
            I = np.clip(I, 0, data1.shape[0] - 1)
            J = np.clip(J, 0, data1.shape[1] - 1)
            line_current.set_ydata(data1[I, J, K, :])
            if chk_autoscale_state.get():
                ax3.relim()        # Recompute limits based on all data (even those out of view)
                ax3.autoscale()

            for cross in ([cross1, cross2, cross3, cross4]):
                update_cross(cross, event.xdata, event.ydata)

            text_coord.set_text(f'I{J} J{I} K{K}')

            if 'mapval4' in globals() and 'data4' in globals():
                mapval4.set_text(f'{data4[I,J,K]:.2f}')
            if 'mapval5' in globals() and 'data5' in globals():
                mapval5.set_text(f'{data5[I,J,K]:.2f}')
            if 'mapval6' in globals() and 'data6' in globals():
                mapval6.set_text(f'{data6[I,J,K]:.2f}')
            plt.draw()

# Keep time series when clicking on image ============================================================
# ====================================================================================================

def on_click_im(event):
    if event.inaxes in (ax1,ax5,ax6,ax4) and event.button == MouseButton.LEFT:
        _,_,K = ijk_anat2func([0,0,slice_n.val])
        J,I,_ = xyz2func_ijk([-event.xdata,-event.ydata,0])

        if 0 <= I < data1.shape[0] and 0 <= J < data1.shape[1] and 0 <= K < data1.shape[2]:
            key = f"{I}_{J}_{K}"  # Construct the key coordinates string
            if aif.mask[I,J,K]:
                aif[I,J,K] = 1
                ove1b.set_data(aif[:,:,K])
                l = ax3.plot(t,data1[I,J,K,:],picker=True, pickradius=3)
                ijk_lines.append([key, l[0]])  # Append the key and plot object pair to the ijk_lines list
            else:
                aif[I,J,K] = np.ma.masked
                ove1b.set_data(aif[:,:,K])
                for item in ijk_lines:
                    if item[0] == key:
                        l = item[1]
                        l.remove()  # Remove the plot object associated with the key
                        ijk_lines.remove(item)  # Remove the item from the ijk_lines list
                        break
            aif_ave = mask_average(data1,aif)

            line_aif_ave.set_ydata(aif_ave)
            line_aif_ave.set_visible(chk_ave_aif_state.get())
            ax3.set_ylim(auto=True)
    plt.draw()
    save_aif()

def load_nifti_filedialog():
    fullfile = filedialog.askopenfilename(initialdir=maindir)
    dir = os.path.dirname(fullfile)
    filename = os.path.basename(fullfile)
    nb_data, data, label = load_nifti(dir,filename)
    window.focus_force()
    return nb_data , data, label

def on_click_im_right(event):
    global data4,ove4,und4,label4,mapval4
    global data5,ove5,und5,label5,mapval5
    global data6,ove6,und6,label6,mapval6
    global data1,ove1,label1,nb_func
    
    if event.button == MouseButton.RIGHT:  
        if event.inaxes == ax1:
            nb_func, data1,label1 = load_nifti_filedialog()
            if 'func_mask' in globals():
                data1 = mask_4d_from_3d(func_mask==0,data1)
            TR = nb_func.header['pixdim'][4]
            resetplot(ax3,data1,TR)
            label1_text.set_text(label1)
            

    if event.button == MouseButton.RIGHT:  
        if event.inaxes == ax4:
            rel_x, rel_y = event.inaxes.transAxes.inverted().transform((event.x, event.y))
            if 0.97 <= rel_x <= 1 and 0.3 <= rel_y <= 0.7:
                x_root = event.guiEvent.x_root
                y_root = event.guiEvent.y_root
                # Show the context menu
                context_menu4.tk_popup(x_root, y_root)
            elif 0.97 <= rel_x <= 1 and 0.15 <= rel_y <= 0.3:
                vmin = simpledialog.askfloat("Input", "Enter new minimum value:", parent=window)
                ove4.set_clim(vmin=vmin)
                apply_cbar_format(ove4.cbar) 
                plt.draw()
            elif 0.97 <= rel_x <= 1 and 0.7 <= rel_y <= 0.85:
                vmax = simpledialog.askfloat("Input", "Enter new maximum value:", parent=window)
                ove4.set_clim(vmax=vmax)
                apply_cbar_format(ove4.cbar) 
                plt.draw()
            else:
                _, newfile_array,label4 = load_nifti_filedialog()
                if 'ove4' in globals():
                    colorscale = ove4.get_cmap()
                else:
                    colorscale = 'hot'
                data4,ove4,und4,mapval4 = show_map(event.inaxes,newfile_array,label4,colorscale,5,5,anat)
    
        elif event.inaxes == ax5:
            rel_x, rel_y = event.inaxes.transAxes.inverted().transform((event.x, event.y))
            if 0.97 <= rel_x <= 1 and 0.3 <= rel_y <= 0.7:
                x_root = event.guiEvent.x_root
                y_root = event.guiEvent.y_root
                # Show the context menu
                context_menu5.tk_popup(x_root, y_root)
            elif 0.97 <= rel_x <= 1 and 0.15 <= rel_y <= 0.3:
                vmin = simpledialog.askfloat("Input", "Enter new minimum value:", parent=window)
                ove5.set_clim(vmin=vmin)
                apply_cbar_format(ove5.cbar) 
                plt.draw()
            elif 0.97 <= rel_x <= 1 and 0.7 <= rel_y <= 0.85:
                vmax = simpledialog.askfloat("Input", "Enter new maximum value:", parent=window)
                ove5.set_clim(vmax=vmax)
                apply_cbar_format(ove5.cbar) 
                plt.draw()    
            else:
                _, newfile_array,label5 = load_nifti_filedialog()
                if 'ove5' in globals():
                    colorscale = ove5.get_cmap()
                else:
                    colorscale = 'hot'
                data5,ove5,und5,mapval5 = show_map(event.inaxes,newfile_array,label5,colorscale,5,5,anat)


        elif event.inaxes == ax6:
            rel_x, rel_y = event.inaxes.transAxes.inverted().transform((event.x, event.y))
            if 0.97 <= rel_x <= 1 and 0.3 <= rel_y <= 0.7:
                x_root = event.guiEvent.x_root
                y_root = event.guiEvent.y_root
                # Show the context menu
                context_menu6.tk_popup(x_root, y_root)
            elif 0.97 <= rel_x <= 1 and 0.15 <= rel_y <= 0.3:
                vmin = simpledialog.askfloat("Input", "Enter new minimum value:", parent=window)
                ove6.set_clim(vmin=vmin)
                apply_cbar_format(ove6.cbar) 
                plt.draw()
            elif 0.97 <= rel_x <= 1 and 0.7 <= rel_y <= 0.85:
                vmax = simpledialog.askfloat("Input", "Enter new maximum value:", parent=window)
                ove6.set_clim(vmax=vmax)
                apply_cbar_format(ove6.cbar) 
                plt.draw()
            else:
                _, newfile_array,label6 = load_nifti_filedialog()
                if 'ove6' in globals():
                    colorscale = ove6.get_cmap()
                else:
                    colorscale = 'hot'
                data6,ove6,und6,mapval6 = show_map(event.inaxes,newfile_array,label6,colorscale,5,5,anat)

    plt.draw()



def update_ylim(val):
    # When dragging, update the y-limits of the plot
    ax3.set_ylim(val)  # Update y-limits with the current slider values
    fig.canvas.draw_idle()  # Redraw the plot

def update_xlim(val):
    # When dragging, update the x-limits of the plot
    ax3.set_xlim(val)  # Update x-limits with the current slider values
    fig.canvas.draw_idle()  # Redraw the plot
       
def reset_slider_limits(event):
    # When mouse is over the slider but the user does not left click on it yet
    if event.inaxes == ylim_slider.ax and event.button != 1:  
        (ymin, ymax) = ax3.get_ylim()
        dy = ymax-ymin
        ylim_slider.valmin = ymin-dy
        ylim_slider.valmax = ymax+dy
        ylim_slider.ax.set_ylim(ymin-dy,ymax+dy)  # Update the slider axis limits
        ylim_slider.set_val([ymin, ymax])  # Update the cursor position
        fig.canvas.draw_idle()  # Redraw the figure

    if event.inaxes == xlim_slider.ax and event.button != 1:  
        (xmin, xmax) = ax3.get_xlim()
        dx = xmax-xmin
        xlim_slider.valmin = xmin-dx
        xlim_slider.valmax = xmax+dx
        xlim_slider.ax.set_xlim(xmin-dx,xmax+dx)  # Update the slider axis limits
        xlim_slider.set_val([xmin, xmax])  # Update the cursor position
        fig.canvas.draw_idle()  # Redraw the figure

def apply_cbar_format(cbar):
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.tick_left()
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color('white')

def apply_cbar_format_new(cbar, grey='0.7', fontsize=13):

    # --- FORCE white background (THIS is the key for inset_axes)
    cbar.ax.patch.set_facecolor('white')
    cbar.ax.patch.set_alpha(1.0)

    # --- ticks + size
    cbar.ax.tick_params(
        axis='y',
        colors=grey,
        labelsize=fontsize
    )

    # --- tick labels: color + bold
    for label in cbar.ax.get_yticklabels():
        label.set_color(grey)
        label.set_fontweight(700)

    # --- colorbar frame
    cbar.outline.set_edgecolor(grey)
    cbar.outline.set_linewidth(1.0)

    cbar.ax.yaxis.tick_left()




def on_menu_selection4(choice):
    ove4.set_cmap(choice)
    apply_cbar_format(ove4.cbar)

def on_menu_selection5(choice):
    ove5.set_cmap(choice)
    apply_cbar_format(ove5.cbar)

def on_menu_selection6(choice):
    ove6.set_cmap(choice)
    apply_cbar_format(ove6.cbar)

# Remove time series when clicking on time series ====================================================
# ====================================================================================================
mousebutton = None
def on_pick_line(event):
    if isinstance(event.artist, plt.Line2D):
        if event.mouseevent.button == MouseButton.LEFT:
            l = event.artist
            l.remove()  # Remove the line from the plot
            for item in ijk_lines:
                if item[1] == l:
                    key = item[0]
                    parts = key.split('_')
                    I, J, K = map(int, parts)  # get the coordinates of the pixel in image aif
                    aif[I, J, K] = np.ma.masked  # Mask pixel in the mask
                    _, _, k_func = ijk_anat2func([0, 0, slice_n.val])
                    ove1b.set_data(aif[:,:,k_func])  # Redisplay the mask without this pixel
                    ijk_lines.remove(item)  # Remove the item from the ijk_lines list
                    break  # Exit the loop after removing the item
            aif_ave = mask_average(data1,aif)
            line_aif_ave.set_ydata(aif_ave) 
            line_aif_ave.set_visible(chk_ave_aif_state.get())
            plt.draw()

        elif event.mouseevent.button == MouseButton.RIGHT:
            l = event.artist
            for item in ijk_lines:
                if item[1] == l:
                    key = item[0]  # Get the key for the file name
                    vox = l.get_ydata()  # Get the y-data (voxel values)
                    parts = key.split('_')
                    I, J, K = map(int, parts)

                    # Open a save file dialog to let the user choose where to save the file
                    save_path = filedialog.asksaveasfilename(
                        initialfile=f"{J}_{I}_{K}.1D",
                        title="Save as",
                        defaultextension=".txt",
                        filetypes=[("Text files", "*.1D")]
                    )

                    if save_path:  # If the user chose a path, save the data
                        np.savetxt(save_path, np.array(vox).transpose())
                        print(f"Saved {key} data to {save_path}")
                    else:
                        print("Save operation canceled.")
    
# Function to load a nifti file
def load_nifti(maindir,filename):
    fullfile = os.path.join(maindir,filename)
    nb_data = load_orient_nifti_afni(fullfile)
    if len(nb_data.shape) == 3:
        data = nb_data.get_fdata().transpose(1,0,2)
    elif len(nb_data.shape) == 4:
        data = nb_data.get_fdata().transpose(1,0,2,3)
    basename, _ = os.path.basename(fullfile).split(".",1)
    if basename == "fwhm": label = "FWHM(s)"
    elif basename == "ttp": label = "TTP(s)"
    elif basename == "bat": label = "BAT(s)"
    elif basename == "cbv": label = "rCBV"
    elif basename == "cbf": label = "rCBF"
    elif basename == "mtt": label = "MTT(s)"
    else: label = basename.upper()
    return nb_data , data, label

# Functions to calculate min and max of colorscale ===================================================
# ====================================================================================================
def vminvmax_percentile_old(array,p1,p2):
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled(np.nan)
    v1 = np.nanpercentile(array, p1)
    v2 = np.nanpercentile(array,100-p2)
    return v1, v2


def vminvmax_percentile(array, p1, p2):
    if isinstance(array, np.ma.MaskedArray):
        array = array.astype(float).filled(np.nan)  # Convert to float before filling NaNs
    else:
        array = array.astype(float)  # Ensure float dtype for regular arrays
    
    v1 = np.nanpercentile(array, p1)
    v2 = np.nanpercentile(array, 100 - p2)
    return v1, v2

def vminvmax_percent(array,perc):
    array_nan = array.filled(np.nan)
    array_nan_1d = array_nan.flatten()
    array_nonan_1d = array_nan_1d[~np.isnan(array_nan_1d)]
    hist, bin_edges = np.histogram(array_nonan_1d, bins='fd')
    y = hist[:-1]
    x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #i for index, h for hight of histogram and v for value of histogram
    perc = perc/100
    imax = np.argmax(y)
    y_perc = max(y)*perc
    iperc_left = np.abs(y[:imax] - y_perc).argmin()
    iperc_right = imax + np.abs(y[imax:] - y_perc).argmin()
    v1 = x[iperc_left]
    v2 = x[iperc_right]
    return v1, v2

def vminvmax_triangle(array,perc):
    array_nan = array.filled(np.nan)
    array_nan_1d = array_nan.flatten()
    array_nonan_1d = array_nan_1d[~np.isnan(array_nan_1d)]
    hist, bin_edges = np.histogram(array_nonan_1d, bins='fd')
    y = hist[:-1]
    x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #i for index, h for hight of histogram and v for value of histogram
    perc = perc/100
    imax = np.argmax(y)
    y_perc = max(y)*perc
    iperc_left = np.abs(y[:imax] - y_perc).argmin()
    iperc_right = imax + np.abs(y[imax:] - y_perc).argmin()
    # Now using Thales, I replace the histogram by a triangle, avoinding outliers and oscilating values on histograms tails ...
    v1 = (x[iperc_left] -perc*x[imax])/(1-perc)
    v2 = (x[iperc_right]-perc*x[imax])/(1-perc)
    return v1, v2

def mask_4d_from_3d(condition_3d, data_4d):

    # Ensure boolean
    condition_3d = condition_3d.astype(bool)

    # Expand to 4D
    condition_4d = condition_3d[..., np.newaxis]
    condition_4d = np.repeat(condition_4d, data_4d.shape[3], axis=3)

    # Existing mask as explicit full-size array
    existing_mask = np.ma.getmaskarray(data_4d)

    # Combine masks
    new_mask = existing_mask | condition_4d

    # Create masked array with explicit mask
    masked_data_4d = np.ma.masked_array(
        np.ma.getdata(data_4d),
        mask=new_mask
    )

    return masked_data_4d

def mask_4d_along_t(data_4d, value_to_mask):
    """
    Create a 4D masked array where the entire t dimension is masked if the value_to_mask
    is present in any of the other dimensions (x, y, z).

    Parameters:
    data_4d (numpy.ndarray): The 4D dataset (shape: x, y, z, t).
    value_to_mask: The value to mask along the t dimension.

    Returns:
    numpy.ma.MaskedArray: The masked 4D array.
    """
    # Check where the value_to_mask is present in the 4D data
    condition_3d = np.any(data_4d == value_to_mask, axis=3)

    # Extend the 3D condition to 4D by adding an extra dimension at the end
    condition_4d = condition_3d[..., np.newaxis]

    # Broadcast the condition to the shape of the 4D data
    condition_4d = np.broadcast_to(condition_4d, data_4d.shape)

    # Create a masked array using the condition
    masked_data_4d = np.ma.masked_where(condition_4d, data_4d)

    return masked_data_4d

def save2nifti(array, affine, form_code, dir, filename):
    global TR
    # Check if the array is a masked array and handle accordingly
    if np.ma.isMaskedArray(array):
        array = array.filled(0)
    
    # Determine the appropriate transpose operation based on the array's dimensions
    if array.ndim == 3:
        array = array.transpose((1, 0, 2))
        # Create the Nifti image
        nb_im = nb.Nifti1Image(array, affine)
    elif array.ndim == 4:
        array = array.transpose((1, 0, 2, 3))
        # Create the Nifti image
        nb_im = nb.Nifti1Image(array, affine)
        nb_im.header['pixdim'][4] = TR
    else:
        raise ValueError("Array must be either 3D or 4D")
    
    # Set the header information
    nb_im.header['sform_code'] = form_code
    nb_im.header['qform_code'] = form_code
    
    # Create the full file path
    fullfile = os.path.join(dir, filename)
    
    # Save the Nifti image
    nb.save(nb_im, fullfile)



# ==================================
# Segmentation function using Kmean
# ===================================
def seg_calc():
    global data4,ove4,und4,mapval4,pr,seg
    global data5,ove5,und5,mapval5

    nb_func_mask = nb.Nifti1Image(func_mask.transpose(1,0,2),aff_func)
    nb_func_mask_rsp = resample_to_img(source_img=nb_func_mask, target_img=nb_anat, interpolation='nearest')
    func_mask_rsp = nb_func_mask_rsp.get_fdata().transpose(1,0,2)  
    
    # Flatten and extract only brain voxels
    anat_mask = func_mask_rsp > 0
    brain_data = anat[anat_mask].reshape(-1, 1)

    # Run KMeans clustering
    opts = read_advanced_options()
    n_clusters = int(opts['seg_n_class'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels_flat = kmeans.fit_predict(brain_data)

    # Assign clustered labels back into full image shape
    labels = np.zeros_like(anat, dtype=np.uint8)
    labels[anat_mask] = labels_flat + 1  # shift to avoid 0 as brain class

    # Optionally sort labels based on intensity to assign CSF/GM/WM
    means = [anat[labels == i].mean() for i in range(1, n_clusters + 1)]
    sorted_idx = np.argsort(means)[::-1]  # high → low

    final_labels = np.zeros_like(labels, dtype=np.uint8)
    for new_val, old_val in enumerate(sorted_idx, start=1):
        final_labels[labels == (old_val + 1)] = new_val

    final_labels_rsp = resample_to_img(source_img=nb.Nifti1Image(final_labels.transpose(1,0,2),nb_anat.affine), target_img=nb_func_mask, interpolation='nearest')
    seg = final_labels_rsp.get_fdata().transpose(1,0,2)

    # Import T1 MNI
    t1_mni = os.path.join(script_directory,'data','T1.nii.gz')
    ants_t1_mni = ants.image_read(t1_mni,reorient='RAI')
    
    # Import the left and right 
    ants_hem_mni = ants.image_read(os.path.join(script_directory,'data','hemispheres.nii.gz'),reorient='RAI')

    # Import anat
    ants_anat = ants.image_read(os.path.join(maindir,anatfullfile),reorient='RAI')
    
    # Import functional
    ants_func = ants.image_read(funcfullfile,reorient='RAI')
    ants_func = ants.slice_image(ants_func, axis=3, idx=0)

    # Registration estimation
    registration = ants.registration(fixed=ants_anat, moving=ants_t1_mni, type_of_transform='AffineFast',aff_metric='mattes')
    
    #Registration application to the right and left hem
    ants_hem = ants.apply_transforms(ants_anat,ants_hem_mni , registration['fwdtransforms'])
    
    
    # Resampling to the functional space
    ants_hem_lowres = ants_hem.resample_image_to_target(ants_func,interp_type='nearestNeighbor').reorient_image2('RAI')
    ants_hem_lowres = ants_hem_lowres.numpy().transpose(1,0,2)
    
    seg = ants_hem_lowres * seg
    


    data4,ove4,und4,mapval4 = show_map(ax4,seg,"GM/WM/CSF",Seg_colors,0,0,anat)
    pr += 1
    save2nifti(seg,aff_func_orig,form_code_func_orig, dir_preprocess, f'func_{pr:02d}_seg.nii.gz')
    plt.draw()



lasso = None  # persistent global
global roi_num
roi_num = [1]
def activate_draw_roi():
    """Toggle between click mode and draw mode"""
    global cid_on_click_im,lasso

    if chk_roi_state.get():
        # ---- DRAW MODE ON ----
        print("Switching to DRAW mode")

        # Disconnect click handler
        fig.canvas.mpl_disconnect(cid_on_click_im)

        # Create lasso on ax4 (or you can change to event.inaxes if you want)
        global lasso
        if lasso is not None:
            lasso.disconnect_events()
            lasso = None
        lasso = LassoSelector(ax4, onlasso, button=1,props=dict(color='red', linewidth=2, alpha=0.8))  # left button
    else:
        # ---- CLICK MODE ON ----
        print("Switching to CLICK mode")

        # Remove any active lasso
        if lasso is not None:
            lasso.disconnect_events()
            lasso = None

        # Reconnect click handler
        cid_on_click_im = fig.canvas.mpl_connect('button_press_event', on_click_im)


def onlasso(vertices):
    # ---- slice index
    _, _, K = ijk_anat2func([0, 0, slice_n.val])
    
    # ---- Convert vertices: anat (x,y) → func (J,I)
    func_vertices = []
    for x, y in vertices:
        J, I, _ = xyz2func_ijk_float([-x, -y, 0])
        func_vertices.append((J, I))
    func_vertices = np.asarray(func_vertices)
    
    if tkvar_roi_type.get() == 'Closed':
        # ---- Build path in FUNCTIONAL space and find points contains in path
        path = Path(func_vertices)
        ny, nx = roi_map.shape[:2]
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.column_stack((x.ravel(), y.ravel()))
        roi_coords = path.contains_points(points,radius=-0.2).reshape(ny, nx)
        
    elif tkvar_roi_type.get() == 'Opened':
        roi_coords = np.zeros_like(roi_map.data[:,:,0],dtype=bool)
        # vertices is a list/array of (J, I) pairs
        for J, I in np.round(func_vertices).astype(int):  # or your converted vertices
            roi_coords[I, J] = True

    # ---- Apply ROI
    if tkvar_roi_num.get() == 'Fill with':
        roi_map[:, :, K][roi_coords] = 1
        roi_map[:] = roi_map[:] * np.ma.masked_where(func_mask==0,func_mask) # Re-mask by the brain mask

    elif tkvar_roi_num.get() == '0':
        roi_map[:, :, K][roi_coords] = np.ma.masked

    else:
        roi_val = int(tkvar_roi_num.get())
        roi_map[:, :, K][roi_coords] = roi_val
        roi_map[:] = roi_map[:] * np.ma.masked_where(func_mask==0,func_mask) # Re-mask by the brain mask

    ove_roi.set_data(roi_map[:, :, K])
    plt.draw()

def load_roi():
    global roi_map, ove_roi
    roi_clean()
    roi = filedialog.askopenfilename(initialdir=maindir,title="Select ROI image (3D)",  filetypes=[("NIFTI files", '*.nii'),("Compressed NIFTI files",'*.gz'),("AFNI files",'*.HEAD')])
    _,roi_map,_ = load_nifti(os.path.dirname(roi),os.path.basename(roi))
    roi_map = np.ma.masked_where(roi_map==0,roi_map)
    _,_,K = ijk_anat2func([0,0,slice_n.val])
    ove_roi.set_data(roi_map[:,:,K])
    window.focus_force()


def roi_clean():
    roi_map[:] = np.ma.masked_all_like(func_mean)
    _,_,K = ijk_anat2func([0,0,slice_n.val])
    ove_roi.set_data(roi_map[:,:,K])
    roi_num[0] = 1
    plt.draw()

def roi_save():
    f = filedialog.asksaveasfilename(initialfile="roi.nii.gz")
    if f:
        save2nifti(roi_map,aff_func_orig,form_code_func_orig, os.path.dirname(f), os.path.basename(f))
    
def roi_ave():
    # Clear and insert new 
    tree_roi.delete(*tree_roi.get_children())
    
    for dir, fname, label in [
        (dir_cvr,'bold.nii.gz', 'BOLD(%)'),
        (dir_cvr,'cvr.nii.gz', 'CVR (%/Δstim)'),
        (dir_cvr,'rcoef.nii.gz', 'RCOEF'),
        (dir_cvr,'cnr.nii.gz', 'CNR'),
        (dir_cvr,'lag.nii.gz', 'LAG(s)'),
        (dir_cvr,'tau.nii.gz', 'TAU(s)'),
        (dir_relative_modfree,'auc.nii.gz', 'AUC'),
        (dir_relative_modfree,'max.nii.gz', 'CMAX'),
        (dir_relative_modfree,'bat.nii.gz', 'BAT(s)'),
        (dir_relative_modfree,'fwhm.nii.gz', 'FWHM(s)'),
        (dir_relative_modfree,'rise.nii.gz', 'T50,ON(s)'),
        (dir_relative_modfree,'decay.nii.gz', 'T50,OFF(s)'),
        (dir_relative_modfree,'ttp.nii.gz', 'TTP(s)'),
        (dir_relative_gamvar,'auc.nii.gz', 'Gamvar AUC'),
        (dir_relative_gamvar,'max.nii.gz', 'GV CMAX'),
        (dir_relative_gamvar,'bat.nii.gz', 'GV BAT(s)'),
        (dir_relative_gamvar,'fwhm.nii.gz', 'GV FWHM(s)'),
        (dir_relative_gamvar,'ttp.nii.gz', 'GV TTP(s)'),
        (dir_relative_gamvar,'rise.nii.gz', 'GV Rise(s)'),
        (dir_relative_gamvar,'decay.nii.gz', 'GV Decay(s)'),
        (dir_quantitative_decon,'cbv.nii.gz', 'rCBV'),
        (dir_quantitative_decon,'mtt.nii.gz', 'MTT(s)'),
        (dir_quantitative_decon,'mtt_cvt.nii.gz', 'MTT_CVT(s)'),
        (dir_quantitative_decon,'Tmax.nii.gz', 'Tmax(s)'),
        (dir_quantitative_decon,'cbf.nii.gz', 'rCBF'),
        (dir_quantitative_decon,'cbf_cvt.nii.gz', 'rCBV/MTT'),
        (dir_quantitative_expon,'mtt.nii.gz', 'Exp MTT(s)'),
        (dir_quantitative_expon,'cbf.nii.gz', 'Exp rCBV/MTT'),
    ]:
        fullfile = os.path.join(dir, fname)
        if os.path.isfile(fullfile):
            _, img, _ = load_nifti(dir, fname)

            row = [label]
            for r in [1,2,3,4,5,6]:
                masked = np.ma.masked_where(roi_map != r, img)

                mean_val = np.ma.mean(masked)
                std_val  = np.ma.std(masked)

                if masked.count() > 0:
                    mean_val = np.round(mean_val, 2)
                    std_val  = np.round(std_val, 2)
                    formatted = f"{mean_val:.1f}±{std_val:.1f}"
                else:
                    formatted = "NA"

                row.append(formatted)

            tree_roi.insert('', 'end', values=row)

            

def copy_roi_table():
    rows = []
    for child in tree_roi.get_children():
        row = tree_roi.item(child)['values']
        rows.append("\t".join(map(str, row)))
    table_text = "Metrics\tROI1\tROI2\tROI3\tROI4\tROI5\tROI6\n" + "\n".join(rows)
    
    # Copy to clipboard
    frame_roi.clipboard_clear()
    frame_roi.clipboard_append(table_text)
    frame_roi.update()  # now it stays on the clipboard after window is closed



#====================================================================================

def set_advanced_options():
    filename = os.path.join(dir_perfmri,'PerfMRI_advanced_options.txt')
    with open(filename, 'w') as file:
        file.write("#========= Pre-processing ==========\n")
        file.write("moco_ref = 0 # The funtional volume# from which the others are realined \n")
        file.write("slice_time_interp = quadratic # Can be linear, cubic, quadratic \n")
        file.write("reg_afni = yes # Uses afni 3dvolreg if yes. Uses NIPY SpaceRealign otherwise \n")
        file.write("dil_mask = 0 # # of voxels dilation around the BOLD mask \n")
        file.write("scale_percentile = 25 # Time series percentile to define baseline for BOLD scaling \n")
        file.write("\n")
        file.write("seg_n_class = 3 # Number of class for segmentation\n")
        file.write("\n")
        file.write("#========= DSC AIF section parameters ==========\n")
        file.write("sr_perc = 20 # Max % Difference in baseline between pre and post bolus\n")
        file.write("sms_perc = 5 # Reject the 5% roughest time series. If AIF results is noisy increase this number  \n")
        file.write("amp_perc = 10 # % of the largest AUC or CMAX \n")
        file.write("amp = cmax # Use either cmax or auc \n")
        file.write("time_para1 = rise # You can use bat, rise, ttp, decay or fwhm \n")
        file.write("time_para2 = ttp # You can use bat, rise, ttp, decay or fwhm \n")
        file.write("time_para3 = decay # You can use bat, rise, ttp, decay or fwhm \n")
        file.write("rho = 1 # Brain density \n")
        file.write("Kh = 0.7 # Hematocrit correction factor \n")
        file.write("\n")
        file.write("#========= CVR AIF section parameters ==========\n")
        file.write("aif_cvr_method = 1 # Method can only be 1 or 2\n")
        file.write("method1_lag_thr = 0.6 # Include voxels whose lag < method1_lag_thr\n")
        file.write("method2_param_order = r lag bold # Metrics order of p-tile thresholding algorithm \n")
        file.write("\n")
        file.write("#========= CVR regression otions ==========\n")
        file.write("pol_deg = 1 #  Num. of polynomial to remove. If 0 only baseline will be remove. If pol_deg=2 baseline, t^1 and t^2 wil be remove\n")
        file.write("use_motparam = no # If yes, use the 6 rigid body mot parameters in the regression \n")
        file.write("\n")
        file.write("npts_gam = 10 # Number of time points to use for the gamma fit  \n")
        file.write("svd_threshold = 20 # Percentage of max value in the inverted AIF matrix under which values are zero out \n")
        
        

def read_advanced_options():
    filename = os.path.join(dir_perfmri,'PerfMRI_advanced_options.txt')
    advanced_options = {}
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace from the beginning and end
            line = line.strip()
            # Ignore empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Split line into variable and value parts
            parts = line.split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                # Extract the value before any comment
                value_part = parts[1].split('#')[0].strip()
                try:
                    # Store it in the dictionary
                    advanced_options[var_name] = value_part
                except ValueError:
                    print(f"Invalid value for {var_name}: {value_part}")

    return advanced_options

def get_options_path(dir_perfmri):
    return os.path.join(dir_perfmri, "PerfMRI_advanced_options.txt")


def load_options(filename):
    with open(filename, "r") as f:
        return f.read()


def save_options(filename, content):
    try:
        with open(filename, "w") as f:
            f.write(content)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file:\n{e}")


def open_advanced_options():

    filename = get_options_path(dir_perfmri)

    win = Toplevel(frame_ui)
    win.title("Advanced Options")
    win.geometry("700x500")

    # --- Text editor ---
    text_widget = Text(win, wrap="word")
    text_widget.pack(expand=True, fill="both")

    # Load file content
    text_widget.insert("1.0", load_options(filename))

    # --- Buttons ---
    btn_frame = Frame(win)
    btn_frame.pack(fill="x")

    def save():
        content = text_widget.get("1.0", "end-1c")
        save_options(filename, content)

    def reset():
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", load_options(filename))

    Button(btn_frame, text="Save", command=save).pack(side="left")
    Button(btn_frame, text="Reset", command=reset).pack(side="left")

# =================================================================================================
# =================================================================================================
# =================================================================================================
# MAKE THE INTERfACE
# =================================================================================================
# =================================================================================================
# =================================================================================================


#Get the current screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
win_geo = '%dx%d'%(screen_height*1.5,screen_height/1.3)
# Adjust window accordingly and make title
window.geometry(win_geo)
window.title("PerfMRI")

# Container that will hold the scrollable canvas + scrollbar
frame_ui_container = Frame(window,width=300)
frame_ui_container.pack(side=LEFT, fill=BOTH,expand=False)


ui_visible = True  # Global toggle flag
pack_opts = dict(side=BOTTOM, fill=BOTH,expand=TRUE)
def toggle_ui():
    global ui_visible
    if ui_visible:
        frame_ui_container.pack_forget()
        toggle_button.config(text="Show UI")
        print("winfo_ismapped:", frame_ui_container.winfo_ismapped())
    else:
        frame_ui_container.pack(**pack_opts)
        frame_ui_canvas.config(scrollregion=frame_ui_canvas.bbox("all"))
        frame_ui_container.lift()
        frame_ui_container.update()
        window.update_idletasks()
        print("size:", frame_ui_container.winfo_width(), frame_ui_container.winfo_height())
        print("Canvas bbox:", frame_ui_canvas.bbox("all"))
        toggle_button.config(text="Hide UI")
    ui_visible = not ui_visible

toggle_button = Button(window, text="Hide UI", command=toggle_ui)
toggle_button.pack(side=TOP)

# Canvas inside container
frame_ui_canvas = Canvas(frame_ui_container,width=frame_ui_canvas_width,bd=0,highlightthickness=0)
frame_ui_canvas.pack(side=LEFT, fill=BOTH, expand=True)


# The actual frame with UI widgets
frame_ui = Frame(frame_ui_canvas)
frame_ui_canvas.create_window((0, 0), window=frame_ui, anchor="nw")

# Update scrollregion when contents grow
def on_configure(event):
    frame_ui_canvas.configure(scrollregion=frame_ui_canvas.bbox("all"))
frame_ui.bind("<Configure>", on_configure)

# Scroll function
def _on_mousewheel(event):
    frame_ui_canvas.yview_scroll(int(-1 * (event.delta / scroll_sensitivity)), "units")

# Bind only when mouse is over the canvas
def _bind_to_mousewheel(event):
    frame_ui_canvas.bind_all("<MouseWheel>", _on_mousewheel)

def _unbind_from_mousewheel(event):
    frame_ui_canvas.unbind_all("<MouseWheel>")

frame_ui.bind("<Enter>", _bind_to_mousewheel)
frame_ui.bind("<Leave>", _unbind_from_mousewheel)



nb_prepro = ttk.Notebook(frame_ui,style="TNotebook")
nb_prepro.pack(side=TOP,anchor="w",fill='both', expand=True)

frame_preprocess = Frame(nb_prepro,pady=20)
nb_prepro.add(frame_preprocess, text='Pre-processings')


# Notebook for MRI and processing info
frame_info = Frame(nb_prepro, pady=20)
nb_prepro.add(frame_info, text='Image Info')
frame_info.rowconfigure(0, weight=1)
frame_info.columnconfigure(0, weight=1)

# Text widget
info_txt = Text(frame_info, wrap="word", width=60, bg="black")
info_txt.grid(row=0, column=0, sticky="nsew")

# Configure tags 
info_txt.tag_configure("info", font=("Helvetica", 14), foreground="white")
info_txt.tag_configure("title", font=("Helvetica", 16, "bold"), foreground="yellow")
info_txt.tag_configure("warn", font=("Helvetica", 14), foreground="red")


nb_analysis = ttk.Notebook(frame_ui,style="TNotebook")
nb_analysis.pack(side=TOP,anchor="w")

# Make a frame for perfusion input
frame_perf = Frame(nb_analysis,pady=20)
nb_analysis.add(frame_perf, text='DSC Analysis')


# Make a frame for CVR input
frame_cvr = Frame(nb_analysis,pady=20)
nb_analysis.add(frame_cvr, text='CVR Analysis')

frame_chkbt = Frame(frame_ui)
frame_chkbt.pack(side=TOP,anchor="w",padx=10, pady=10)


# ==== ROI FRAME ======================================================================
#======================================================================================

# Make a frame for ROI frame
frame_roi = Frame(nb_analysis,pady=20)
#nb_analysis.add(frame_seg, text='Segmentation')
nb_analysis.add(frame_roi, text='ROI')

# Make 2 frames one for the ROI controls and one for the results table
frame_roi_controls = Frame(frame_roi)
frame_roi_controls.grid(row=2, column=0, columnspan=6, sticky=W)

frame_roi_table = Frame(frame_roi)
frame_roi_table.grid(row=3, column=0, columnspan=6, sticky=W)

# Load ROI
lb_load_roi = Label(frame_roi_controls, text="Load ROI")
lb_load_roi.grid( row=1,column=0,sticky=W, padx=(10,0))

bnt_load_roi = Button(frame_roi_controls, text="File", justify='center',command=load_roi,width=bt_small,font=bt_font)
bnt_load_roi.grid(row=1, column=1, sticky=W, padx=(10,0))

# Draw ROI
lb_draw_roi = Label(frame_roi_controls, text="Draw ROI")
lb_draw_roi.grid(row=2,column=0,sticky=W, padx=(10,0))

# Insert a checkbutton to activate ROI drawing
chk_roi_state = BooleanVar()
chk_roi_state.set(False) #set check state
chk_roi = Checkbutton(frame_roi_controls, text='',command=activate_draw_roi,variable=chk_roi_state)
chk_roi.grid(row=2,column=1, sticky=W,padx=(10,0))

# ROI type
tkvar_roi_type = StringVar()
choices_roi_type = ['Closed','Opened']
menu_roi_type = OptionMenu(frame_roi_controls, tkvar_roi_type, *choices_roi_type)
tkvar_roi_type.set('Closed') # set the default option
menu_roi_type.grid(row=2,column=3,sticky=W,padx=(10,0))
menu_roi_type.config(width=4)

# ROI number
tkvar_roi_num = StringVar()
choices_roi_num = ['Fill with','0','1','2','3 ','4','5','6']
menu_roi_num = OptionMenu(frame_roi_controls, tkvar_roi_num, *choices_roi_num)
tkvar_roi_num.set('Fill with') # set the default option
menu_roi_num.grid(row=2,column=4,sticky=W,padx=(10,0))
menu_roi_num.config(width=4)

bnt_roi = Button(frame_roi_controls, text="save", justify='center',command=roi_save,width=bt_small,font=bt_font)
bnt_roi.grid(row=2,column=5,sticky=W, padx=(10,0))

bnt_roi = Button(frame_roi_controls, text="clean", justify='center',command=roi_clean,width=bt_small,font=bt_font)
bnt_roi.grid(row=2,column=6,sticky=W, padx=(10,0))

# Calc average metrics within ROIs
lb_calc_roi = Label(frame_roi_controls, text="Ave. ROI")
lb_calc_roi.grid( row=3,column=0,sticky=W, padx=(10,0))

bnt_calc_roi = Button(frame_roi_controls, text="calc", justify='center',command=roi_ave,width=bt_small,font=bt_font)
bnt_calc_roi.grid(row=3,column=1,sticky=W, padx=(10,0))

bt_copy = Button(frame_roi_controls, text="copy",command=copy_roi_table,width=bt_small,font=bt_font)
bt_copy.grid(column=3, row=3,sticky=W,padx=(10,0))

#  Create tree_roi  table ===========================================


tree_roi = ttk.Treeview(
    frame_roi_table, 
    columns=('metric', 'roi1', 'roi2', 'roi3', 'roi4','roi5','roi6'), 
    show='headings', 
    height=8
)

#s = ttk.Style()
#s.configure('Treeview.Heading', background='black', foreground='dark blue')


# Set up the column headings
tree_roi.heading('metric', text='Metric')
tree_roi.heading('roi1', text="🟥ROI1")
tree_roi.heading('roi2', text="🟩ROI2")
tree_roi.heading('roi3', text="🟦ROI3")
tree_roi.heading('roi4', text="🟧ROI4")
tree_roi.heading('roi5', text="🟪ROI5")
tree_roi.heading('roi6', text="🟨ROI6")

# Set up the column widths and alignment
tree_roi.column('metric', width=int(metric_width), anchor='w')
tree_roi.column('roi1', width=value_width, anchor='center')
tree_roi.column('roi2', width=value_width, anchor='center')
tree_roi.column('roi3', width=value_width, anchor='center')
tree_roi.column('roi4', width=value_width, anchor='center')
tree_roi.column('roi5', width=value_width, anchor='center')
tree_roi.column('roi6', width=value_width, anchor='center')
tree_roi.grid(row=0,column=0,sticky=W, padx=(10,0),pady=(20,0),columnspan=6)


tree_roi.tag_configure('red_col', foreground='red')
tree_roi.tag_configure('green_col', foreground='green')
tree_roi.tag_configure('blue_col', foreground='blue')
tree_roi.tag_configure('orange_col', foreground='orange')
tree_roi.tag_configure('purple_col', foreground='purple')
tree_roi.tag_configure('yellow_col', foreground='#b58900')


# Make a frame for Segmentation
frame_seg = Frame(nb_analysis,pady=20)
nb_analysis.add(frame_seg, text='Segmentation')

# Create the Treeview in your frame
tree = ttk.Treeview(
    frame_seg, 
    columns=('metric', 'rgm', 'rwm', 'lgm', 'lwm'), 
    show='headings', 
    height=8
)

# Set up the column headings
tree.heading('metric', text='Metric')
tree.heading('rgm', text='R GM')
tree.heading('rwm', text='R WM')
tree.heading('lgm', text='L GM')
tree.heading('lwm', text='L WM')

# Set up the column widths and alignment
seg_width = 80
tree.column('metric', width=metric_width, anchor='w')
tree.column('lgm', width=seg_width, anchor='center')
tree.column('lwm', width=seg_width, anchor='center')
tree.column('rgm', width=seg_width, anchor='center')
tree.column('rwm', width=seg_width, anchor='center')

# Place the Treeview widget in the layout
tree.grid(row=5, column=0, columnspan=4, sticky=W, padx=(10, 40), pady=10)
# Updated seg_ave function
def seg_ave():
    
    rows = []
    for dir, fname, label in [
        (dir_cvr,'bold.nii.gz', 'BOLD(%)'),
        (dir_cvr,'cvr.nii.gz', 'CVR (%/Δstim)'),
        (dir_cvr,'rcoef.nii.gz', 'RCOEF'),
        (dir_cvr,'CNR.nii.gz', 'CNR'),
        (dir_cvr,'lag.nii.gz', 'LAG(s)'),
        (dir_cvr,'tau.nii.gz', 'TAU(s)'),
        (dir_relative_modfree,'auc.nii.gz', 'AUC'),
        (dir_relative_modfree,'max.nii.gz', 'CMAX'),
        (dir_relative_modfree,'bat.nii.gz', 'BAT(s)'),
        (dir_relative_modfree,'fwhm.nii.gz', 'FWHM(s)'),
        (dir_relative_modfree,'ttp.nii.gz', 'TTP(s)'),
        (dir_relative_gamvar,'auc.nii.gz', 'Gamvar AUC'),
        (dir_relative_gamvar,'max.nii.gz', 'GV CMAX'),
        (dir_relative_gamvar,'bat.nii.gz', 'GV BAT(s)'),
        (dir_relative_gamvar,'fwhm.nii.gz', 'GV FWHM(s)'),
        (dir_relative_gamvar,'ttp.nii.gz', 'GV TTP(s)'),
        (dir_quantitative_decon,'cbv.nii.gz', 'rCBV'),
        (dir_quantitative_decon,'mtt.nii.gz', 'MTT(s)'),
        (dir_quantitative_decon,'Tmax.nii.gz', 'Tmax(s)'),
        (dir_quantitative_decon,'cbf.nii.gz', 'rCBF'),
        (dir_quantitative_decon,'cbf_cvt.nii.gz', 'rCBV/MTT'),
        (dir_quantitative_expon,'mtt.nii.gz', 'Exp MTT(s)'),
        (dir_quantitative_expon,'cbf.nii.gz', 'Exp rCBV/MTT'),
        
    ]:
        fullfile = os.path.join(dir, fname)
        if os.path.isfile(fullfile):
            _, img, _ = load_nifti(dir, fname)
            rgm_val = np.round(np.ma.mean(np.ma.masked_where(seg != 2, img)), 2)
            rwm_val = np.round(np.ma.mean(np.ma.masked_where(seg != 1, img)), 2)
            lgm_val = np.round(np.ma.mean(np.ma.masked_where(seg != 20, img)), 2)
            lwm_val = np.round(np.ma.mean(np.ma.masked_where(seg != 10, img)), 2)
            rows.append((label, rgm_val, rwm_val, lgm_val, lwm_val)) 


        # Clear and insert new 
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert('', 'end', values=row)


def copy_tree_to_clipboard():
    rows = []
    for child in tree.get_children():
        row = tree.item(child)['values']
        rows.append("\t".join(map(str, row)))
    table_text = "Metrics\tR GM\tR WM\tL GM\tL WM\n" + "\n".join(rows)
    
    # Copy to clipboard
    frame_seg.clipboard_clear()
    frame_seg.clipboard_append(table_text)
    frame_seg.update()  # now it stays on the clipboard after window is closed

bt_func = Button(frame_seg, text="seg",command=seg_calc,width=bt_small,font=bt_font)
bt_func.grid(column=0, row=2,sticky=W,padx=(10,0))

bt_anat = Button(frame_seg, text="avg",command=seg_ave,width=bt_small,font=bt_font)
bt_anat.grid(column=0, row=2,sticky=W,padx=(60,0))

bt_copy = Button(frame_seg, text="copy",command=copy_tree_to_clipboard,width=bt_small,font=bt_font)
bt_copy.grid(column=0, row=2,sticky=W,padx=(110,0))





# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================


# Pre-Process Export Window
def open_preprocess_export_window():
    window_preprocess = Toplevel(window)
    window_preprocess.title("Pre-Process Files")
    
    pre_vars = [
        ("Mask Brain", "mas", BooleanVar(value=False)),
        ("Trim Signal","tri", BooleanVar(value=False)),
        ("Time Realign","tre", BooleanVar(value=False)),
        ("Space Realign","sre", BooleanVar(value=False)),
        ("Detrend Signal","dtr", BooleanVar(value=False)),
        ("Scale Signal","scl", BooleanVar(value=False)),
        ("Spatial Smoothing","ssm",BooleanVar(value=False)),
        ("Final preprocessed","final", BooleanVar(value=True)),
    ]



    def export_preprocess_files():
        folder_selected = filedialog.askdirectory(initialdir=maindir,title="Select or Create Folder to Save Files")
        if not folder_selected:
            print("Export canceled.")
            return

        all_files_in_prepro = os.listdir(dir_preprocess)

        for text, keyword, var in pre_vars:
            if var.get():
                matched_files = [f for f in all_files_in_prepro if keyword in f]
                for f in matched_files:
                    src = os.path.join(dir_preprocess, f)
                    dst = os.path.join(folder_selected, f)
                    shutil.copy(src, dst)
                    print(f"Copied: {f}")

                if keyword == "final":
                    sorted_files = sorted(all_files_in_prepro, key=lambda f: os.path.getmtime(os.path.join(dir_preprocess, f)))
                    latest_file = sorted_files[-1]
                    src = os.path.join(dir_preprocess, latest_file)
                    dst = os.path.join(folder_selected, latest_file)
                    shutil.copy(src, dst)
                    print(f"Copied latest file: {latest_file}")

        window_preprocess.destroy()

    all_files_in_prepro = os.listdir(dir_preprocess)
    Label(window_preprocess,text="").grid(column=1,row=0,pady=10,padx=10)
    for i, (text,keyword,var) in enumerate(pre_vars):
        c = Checkbutton(window_preprocess, text=text, variable=var)
        c.grid(column=0, row=i+2, sticky=W, padx=(10,0), pady=(5,0))
        matched_files = [f for f in all_files_in_prepro if keyword in f]
        if not matched_files:
            c.config(state='disabled') 
    c.config(state='normal') # Enable the last checkbutton as it is the final preprocess
    b = Button(window_preprocess, text="Export ...", command=export_preprocess_files,fg="blue")
    b.grid(column=2,row=40, sticky=W, padx=(10,0),pady=(40,10))    

# Perfusion Export Window
def open_perfusion_export_window():
    window_perfusion = Toplevel(window)
    window_perfusion.title("Export Perfusion Files")

    perfusion_vars_col1 = [
        ("AUC", "auc", BooleanVar(value=True)),
        ("Max", "cmax", BooleanVar(value=True)),
        ("TTP", "ttp", BooleanVar(value=True)),
        ("FWHM", "fwhm", BooleanVar(value=True)),
        ("BAT", "bat", BooleanVar(value=True)),
        ("Rise", "rise", BooleanVar(value=True)),
        ("Decay", "decay", BooleanVar(value=True)),
    ]

    perfusion_vars_col2 = [
        ("AUC", "auc", BooleanVar(value=True)),
        ("Max", "cmax", BooleanVar(value=True)),
        ("TTP", "ttp", BooleanVar(value=True)),
        ("FWHM", "fwhm", BooleanVar(value=True)),
        ("BAT", "bat", BooleanVar(value=True)),
        ("Rise", "rise", BooleanVar(value=True)),
        ("Decay", "decay", BooleanVar(value=True)),
    ]

    perfusion_vars_col_aif = [
        ("Mask AIF", "aif", BooleanVar(value=True)),
        ("Mean AIF", "mean", BooleanVar(value=True)),
        ("Multi AIF", "multi", BooleanVar(value=True)),
    ]

    perfusion_vars_col3 = [
        ("CBV", "cbv", BooleanVar(value=True)),
        ("CBF", "cbf", BooleanVar(value=True)),
        ("MTT", "mtt", BooleanVar(value=True)),
        ("Tmax", "Tmax", BooleanVar(value=True)),
        ("Residue Function", "R", BooleanVar(value=True)),
    ]

    perfusion_vars_col4 = [
    ("CBV", "cbv", BooleanVar(value=True)),
    ("CBF", "cbf", BooleanVar(value=True)),
    ("MTT", "mtt", BooleanVar(value=True)),
    ]


    def export_perfusion_files():
        folder_selected = filedialog.askdirectory(initialdir=maindir,title="Select or Create Folder to Save Files")
        if not folder_selected:
            print("Export canceled.")
            return

        if os.path.isdir(dir_relative_modfree):
            all_files_in_modfree = os.listdir(dir_relative_modfree)
            for text, keyword, var in perfusion_vars_col1 + perfusion_vars_col_aif:
                if var.get():
                    matched_files = [f for f in all_files_in_modfree if keyword in f]
                    for f in matched_files:
                        src = os.path.join(dir_relative_modfree, f)
                        basename = os.path.basename(f) 
                        dst = os.path.join(folder_selected,basename + "__modfree.nii.gz")
                        shutil.copy(src, dst)

        if os.path.isdir(dir_relative_gamvar):
            all_files_in_gamvar = os.listdir(dir_relative_gamvar)
            for text, keyword, var in perfusion_vars_col2 + perfusion_vars_col_aif:
                if var.get():
                    matched_files = [f for f in all_files_in_modfree if keyword in f]
                    for f in matched_files:
                        src = os.path.join(dir_relative_modfree, f)
                        basename = os.path.basename(f) 
                        dst = os.path.join(folder_selected, basename + "__gamvar.nii.gz")
                        shutil.copy(src, dst)

        if os.path.isdir(dir_quantitative_decon):
            all_files_in_quantitative_decon = os.listdir(dir_quantitative_decon)
            for text, keyword, var in perfusion_vars_col3:
                if var.get():
                    matched_files = [f for f in all_files_in_quantitative_decon if keyword in f]
                    for f in matched_files:
                        src = os.path.join(dir_quantitative_decon, f)
                        basename = os.path.basename(f) 
                        dst = os.path.join(folder_selected, basename + "__svd.nii.gz")
                        shutil.copy(src, dst)

        if os.path.isdir(dir_quantitative_expon):
            all_files_in_quantitative_expon = os.listdir(dir_quantitative_expon)
            for text, keyword, var in perfusion_vars_col4:
                if var.get():
                    matched_files = [f for f in all_files_in_quantitative_expon if keyword in f]
                    for f in matched_files:
                        src = os.path.join(dir_quantitative_expon, f)
                        basename = os.path.basename(f) 
                        dst = os.path.join(folder_selected, basename + "__exp.nii.gz")
                        shutil.copy(src, dst)


        
        
        window_perfusion.destroy()
    
    
    # Column 0
    Label(window_perfusion, text="Model Free\nParameters\n",justify=LEFT).grid(column=0, row=0, sticky=NW, padx=(20, 0), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(perfusion_vars_col1):
        if not os.path.isdir(dir_relative_modfree) or not any(keyword in f for f in os.listdir(dir_relative_modfree)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_perfusion, text=text, variable=var, state=state).grid(
            column=0, row=i+1, sticky=W, padx=(20, 0), pady=(5, 0)
        )

    # Column 1
    Label(window_perfusion, text="Gamma Var\nParameters\n",justify=LEFT).grid(column=1, row=0, sticky=NW, padx=(40, 0), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(perfusion_vars_col2):
        if not os.path.isdir(dir_relative_gamvar) or not any(keyword in f for f in os.listdir(dir_relative_gamvar)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_perfusion, text=text, variable=var, state=state).grid(
            column=1, row=i+1, sticky=W, padx=(40, 0), pady=(5, 0)
        )

    # Column 2 – Arterial Input Function
    Label(window_perfusion, text="Arterial Input\nFunction\n",justify=LEFT).grid(column=2, row=0, sticky=NW, padx=(40, 0), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(perfusion_vars_col_aif):
        if not os.path.isdir(dir_relative_modfree) or not any(keyword in f for f in os.listdir(dir_relative_modfree)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_perfusion, text=text, variable=var, state=state).grid(
            column=2, row=i+1, sticky=W, padx=(40, 0), pady=(5, 0)
        )

    # Column 3 – Quantitative Measures (SVD)
    Label(window_perfusion, text="Quantitative\nMeasures\n(SVD)",justify=LEFT).grid(column=3, row=0, sticky=NW, padx=(40, 0), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(perfusion_vars_col3):
        if not os.path.isdir(dir_quantitative_decon) or not any(keyword in f for f in os.listdir(dir_quantitative_decon)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_perfusion, text=text, variable=var, state=state).grid(
            column=3, row=i+1, sticky=W, padx=(40, 0), pady=(5, 0)
        )

    # Column 4 – Quantitative Measures (Exp)
    Label(window_perfusion, text="Quantitative\nMeasures\n(Exp)",justify=LEFT).grid(column=4, row=0, sticky=NW, padx=(40, 10), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(perfusion_vars_col4):
        if not os.path.isdir(dir_quantitative_expon) or not any(keyword in f for f in os.listdir(dir_quantitative_expon)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_perfusion, text=text, variable=var, state=state).grid(
            column=4, row=i+1, sticky=W, padx=(40, 10), pady=(5, 0)
        )

    # Export Button
    b = Button(window_perfusion, text="Export ...", command=export_perfusion_files, fg="blue")
    b.grid(column=0, row=40, sticky=W, padx=(10, 0), pady=(40, 10))


# CVR Export Window
def open_cvr_export_window():
    window_cvr = Toplevel(window)
    window_cvr.title("Export CVR Files")
    
    cvr_vars_left = [
        ("CVR Map", 'cvr.nii.gz', BooleanVar(value=True)),
        ("Correlation Map", 'rcoef.nii.gz', BooleanVar(value=True)),
        ("Lag Map", 'lag.nii.gz', BooleanVar(value=True)),
        ("CNR Map", 'cnr.nii.gz', BooleanVar(value=True)),
        ("CNR Boluses", 'cnr_boluses.nii.gz', BooleanVar(value=True)),
        
    ]

    cvr_vars_right = [
        ("Shifted Ref","ref_shifted.1D", BooleanVar(value=True)),
        ("AIF Mask",'aif.nii.gz', BooleanVar(value=True)),
        ("Mean AIF Curve (+)",'mean_AIF_pos.1D',BooleanVar(value=True)),
        ("Mean AIF Curve (-)",'mean_AIF_neg.1D',BooleanVar(value=True)),
        ("Multi AIF Curves (+)", 'multi_AIF_pos.1D' ,BooleanVar(value=True)),
        ("Multi AIF Curves (-)", 'multi_AIF_neg.1D' ,BooleanVar(value=True)),
    ]
    
    
    

    def export_cvr_files():
        folder_selected = filedialog.askdirectory(initialdir=maindir,title="Select or Create Folder to Save Files")
        if not folder_selected:
            print("Export canceled.")
            return

        if os.path.isdir(dir_relative_modfree):
            all_files_in_cvr = os.listdir(dir_cvr)
            for text, keyword, var in cvr_vars_left + cvr_vars_right:
                if var.get():
                    matched_files = [f for f in all_files_in_cvr if keyword in f]
                    for f in matched_files:
                        src = os.path.join(dir_cvr, f)
                        basename = os.path.basename(f) 
                        dst = os.path.join(folder_selected,basename + "__cvr.nii.gz")
                        shutil.copy(src, dst)

        window_cvr.destroy()

    
    # Column left
    Label(window_cvr, text="Vascular\nReactivity\n",justify=LEFT).grid(column=0, row=0, sticky=NW, padx=(20, 0), pady=(10, 10))
    for i, (text, keyword, var) in enumerate(cvr_vars_left):
        if not os.path.isdir(dir_cvr) or not any(keyword in f for f in os.listdir(dir_cvr)):
            state = 'disable'
        else:
            state = 'normal'
        Checkbutton(window_cvr, text=text, variable=var, state=state).grid(
            column=0, row=i+1, sticky=W, padx=(40, 10), pady=(5, 0)
        )
    # Export Button
    b = Button(window_cvr, text="Export ...", command=export_cvr_files, fg="blue")
    b.grid(column=0, row=40, sticky=W, padx=(10, 0), pady=(40, 10))

# --- Buttons in the different frames of the UI

bt_export_preprocess = Button(frame_preprocess, text="Export ...", command=open_preprocess_export_window, fg="blue")
bt_export_preprocess.grid(column=1,row=40, sticky=W, padx=10,pady=(10,0))

bt_export_perfusion = Button(frame_perf, text="Export ...", command=open_perfusion_export_window, fg="blue")
bt_export_perfusion.grid(column=0,row=40, sticky=W, padx=10, pady=10)

bt_export_cvr = Button(frame_cvr, text="Export ...", command=open_cvr_export_window, fg="blue",bg="grey")
bt_export_cvr.grid(column=0,row=40, sticky=W, padx=10, pady=10)



# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================



current_tab_index = nb_analysis.index("current")

    # Get the text of the currently selected tab
current_tab_text = nb_analysis.tab(current_tab_index, "text")

bt_adv_opt = Button(frame_ui, text="AOW",command=open_advanced_options,fg="blue")
bt_adv_opt.pack(side=TOP,anchor="w",padx=10, pady=10)


# Create a Frame for advanced options
frame_opts=Frame(frame_ui)
frame_opts.pack(side=TOP,anchor="w")

# Create a Text widget for editing the options
text_widget = Text(frame_opts, wrap='word', width=70, height=20,highlightcolor="blue")
text_widget.pack()


# Save button
save_button = Button(frame_opts, text="Save", command=save_options)
save_button.pack(side=LEFT)

# Reset button
reset_button = Button(frame_opts, text="Reset", command=set_advanced_options)
reset_button.pack(side=LEFT)

frame_opts.pack_forget()



# Make a frame for messages at bottom
frame_path = Frame(window)
frame_path.pack(side=BOTTOM, padx=10, pady=10)
anatdir = Label(frame_path, text="Anatomical:",fg="white",bg='black',relief='flat')
anatdir.grid(column=0, row=0,sticky=W,padx=(10,0),columnspan=15)
funcdir  = Label(frame_path, text='Functional: ', fg="white",bg='black',relief='flat')
funcdir.grid(column=0, row=1,sticky=W,padx=(10,0),columnspan=15)




# Setup the rows numbers for the first panel
row_dicom2nifti = 1
row_input_nifti = 2
row_input_anat = 3
row_slicetime = 4
row_space_realign = 5
row_scale = 6

# lb means label
# bt means button

# Setup AFNI script path if you use the gammavar fit model to calculate rel. perfusion
script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)


#  TAB frame_perf
# LINE 01
# Convert files from dicom to nifti
lb_convert = Label(frame_preprocess, text="Dicom2nifti")
lb_convert.grid(column=1, row=2,sticky=W,padx=(10,40))
bt_func = Button(frame_preprocess, text="func",command=convert_func,width=bt_small)
bt_func.grid(column=2, row=2,sticky=W)
def convert_anatt():
    print("func_mask ===============")
    np_info(func_mask)
    print("=========================")
    print("data1====================")
    np_info(data1)
    print("=========================")

bt_anat = Button(frame_preprocess, text="anat",command=convert_anatt,width=bt_small,font=bt_font)
bt_anat.grid(column=3, row=2,sticky=W)

# LINE 02
# Choose FUNC and ANAT files
lb_inp_nifti = Label(frame_preprocess, text="Input nifti")
lb_inp_nifti.grid(column=1, row=3,sticky=W,padx=(10,40))

bnt_choose_funcfile = Button(frame_preprocess, text="func",command=click_choose_funcfile,width=bt_small,font=bt_font)
bnt_choose_funcfile.grid(column=2, row=3,sticky=W)

bnt_choose_anatfile = Button(frame_preprocess, text="anat",command=click_choose_anatfile,width=bt_small,font=bt_font)
bnt_choose_anatfile.grid(column=3, row=3,sticky=W)

bt_load_raw = Button(frame_preprocess, text="load",command=press_load_raw,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_load_raw.grid(column=4, row=3,sticky=W)

# Trim the signal
lb_trim_signal = Label(frame_preprocess, text="Trim Signal")
lb_trim_signal.grid(column=1, row=4,sticky=W,padx=(10,40))

bnt_set_vlines = Button(frame_preprocess, text="set", justify='center',command=set_vlines,width=bt_small,font=bt_font)
bnt_set_vlines.grid(column=2, row=4, sticky=W)

bnt_trim_signal = Button(frame_preprocess, text="calc",command=press_trim_signal,width=bt_small,font=bt_font)
bnt_trim_signal.grid(column=3, row=4,sticky=W)


# Slice Time correction
lb_slicetime = Label(frame_preprocess, text="Time   Realign")
lb_slicetime.grid(column=1, row=5,sticky=W,padx=(10,10))

# Choose slice time ordering
tkvar_slicetime = StringVar()
choices_slicetime = ['Alternative+','Sequential+','Text file...']
menu_slicetime = OptionMenu(frame_preprocess, tkvar_slicetime, *choices_slicetime)
tkvar_slicetime.set('Alternative+') # set the default option
menu_slicetime.grid(row=5, column=2, sticky=W,columnspan=2)
menu_slicetime.config(width=8)

# Attach the trace to the StringVar, monitoring changes
tkvar_slicetime.trace('w', on_slicetime_change)
bt_slicetime = Button(frame_preprocess, text="calc",command=press_time_realign,width=bt_small,font=bt_font)
bt_slicetime.grid(column=4, row=5,sticky=W)


# Volume re-registration
lb_space_realign = Label(frame_preprocess, text="Space Realign")
lb_space_realign.grid(column=1, row=6,sticky=W,padx=(10,10))

# Calc 
bt_space_realign = Button(frame_preprocess, text="calc",command=press_space_realign,width=bt_small,font=bt_font)
bt_space_realign.grid(column=2, row=6,sticky=W)

# View motion parameters 
bt_view_realign = Button(frame_preprocess, text="view",command=press_view_motion,width=bt_small,font=bt_font)
bt_view_realign.grid(column=3, row=6,sticky=W)

# Register anat to func
# lb_anat2func = Label(frame_preprocess, text="Reg anat to func")
# lb_anat2func.grid(column=0, row=7,sticky=W,padx=(10,10))
# bt_anat2func = Button(frame_preprocess, text="calc",command=press_reg_anat2func,width=bt_small,font=bt_font)
# bt_anat2func.grid(column=1, row=7,sticky=W)

# Spatial smoothing
lb_spatial_smoothing = Label(frame_preprocess, text="Smooth Space")
lb_spatial_smoothing.grid(column=1, row=8, sticky=W, padx=(10,10))

lb_fwhm = Label(frame_preprocess, text="FWHM (mm)")
lb_fwhm.grid(column=2, row=8, sticky=W,columnspan=2)

sb_fwhm = Spinbox(frame_preprocess, from_=0, to=20,increment=0.1,width=3)  # You can set a range, like 0 to 20
sb_fwhm.grid(row=8, column = 4, sticky=W)

bt_spatial_smoothing = Button(frame_preprocess, text="calc", justify='center',command=press_spatial_smoothing,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_spatial_smoothing.grid(column=5, row=8, sticky=W)

# Temporal smoothing
lb_temporal_smoothing = Label(frame_preprocess, text="Smooth Time")
lb_temporal_smoothing.grid(column=1, row=9, sticky=W, padx=(10,10))

lb_fwhm_t = Label(frame_preprocess, text="FWHM(s)")
lb_fwhm_t.grid(column=2, row=9, sticky=W,columnspan=2)

sb_fwhm_t = Spinbox(frame_preprocess, from_=0, to=20,increment=0.1,width=3)  # You can set a range, like 0 to 20
sb_fwhm_t.grid(row=9, column = 4, sticky=W)

bt_temporal_smoothing = Button(frame_preprocess, text="calc", justify='center',command=press_temporal_smoothing,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_temporal_smoothing.grid(column=5, row=9, sticky=W)

# Mask brain
lb_mask_brain = Label(frame_preprocess, text="Mask Brain")
lb_mask_brain.grid(column=1, row=10, sticky=W, padx=(10,40))

# Choose Masking option
tkvar_masktype = StringVar()
choices_masktype = ['Automask','Use Zeros']
menu_masktype = OptionMenu(frame_preprocess, tkvar_masktype, *choices_masktype)
tkvar_masktype.set('Automask') # set the default option
menu_masktype.grid(column = 2,row=10,sticky=W,columnspan=2)
menu_masktype.config(width=8)

bnt_mask_brain = Button(frame_preprocess, text="calc", justify='center',command=press_mask_brain,width=bt_small,font=bt_font)
bnt_mask_brain.grid(column=4, row=10, sticky=W)


# Detrend signal
lb_detrend = Label(frame_preprocess, text="Drift Correction")
lb_detrend.grid(column=1, row=11,sticky=W,padx=(10,40))

# Polynomial
tkvar_poly = StringVar()
choices_poly = ['Polynomial #','1','2','3 ','4','5','6','7','8','9','10','11','12']
menu_poly = OptionMenu(frame_preprocess, tkvar_poly, *choices_poly)
tkvar_poly.set('Polynomial #') # set the default option
menu_poly.grid(row=11,column=2,sticky=W,columnspan=2)
menu_poly.config(width=9)


bt_detrend = Button(frame_preprocess, text="calc",command=press_detrend_signal,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_detrend.grid(column=4, row=11,sticky=W)


#Scale signal
lb_scale = Label(frame_preprocess, text="Scale signal")
lb_scale.grid(column=1, row=13,sticky=W,padx=(10,10))

# Choose between Signal or Concentration time series
tkvar_imtype = StringVar()
choices_imtype = ['Concentration','% baseline','% mean',"% Percentile"]
menu_imtype = OptionMenu(frame_preprocess, tkvar_imtype, *choices_imtype)
tkvar_imtype.set('Concentration') # set the default option
menu_imtype.grid(row=13, column = 2, sticky=W,columnspan=2)
menu_imtype.config(width=9)

bt_scale_signal = Button(frame_preprocess, text="calc",command=press_scale_signal,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_scale_signal.grid(column=4, row=13,sticky=W)

# Undo last pre-processing step
lb_undo = Label(frame_preprocess, text="Undo last step")
lb_undo.grid(column=1, row=14,sticky=W,padx=(10,10))

lb_undo = Button(frame_preprocess, text="↶",command=undo_prepro,width=bt_small,font=bt_font,fg="black",activeforeground="red")
lb_undo.grid(column=2, row=14,sticky=W)


# LINE 06
# Relative perfusion

lb_relperf = Label(frame_perf, text="Relative Perfusion",justify='left')
lb_relperf.grid(row=0,column=0,sticky=W,padx=(10,40))

relperf_method = StringVar()
choices = ['Model Free','GamVar-afni']
protocol_menu = OptionMenu(frame_perf, relperf_method, *choices)
relperf_method.set('Model Free') # set the default option
protocol_menu.grid(row=0, column=1, sticky=W, columnspan=2, padx=(0,0))
protocol_menu.config(width=8)

bt_calc_relperf = Button(frame_perf, text="calc",command=press_calc_relperf,width=bt_small,font=bt_font,activeforeground="red")
bt_calc_relperf.grid(column=3, row=0,sticky=W, padx=(0,0))

bt_view_relperf = Button(frame_perf, text="view",command=view_relperf,width=bt_small,font=bt_font,fg="black")
bt_view_relperf.grid(column=4, row=0,sticky=W, padx=(0,0))

# This was the line to correct the time map with time slices instead of the slice time correction on the raw dataset
# lb_corr_time = Label(frame_perf, text="Correct time maps",justify='left')
# lb_corr_time.grid(row=1,column=0,sticky=W,padx=(10,40))

# # Choose slice time ordering
# tkvar_correct_time = StringVar()
# choices_correct_time = ['Alternative+','Sequential+','Text file...']
# menu_correct_time = OptionMenu(frame_perf, tkvar_correct_time, *choices_correct_time)
# tkvar_correct_time.set('Alternative+') # set the default option
# menu_correct_time.grid(row=1, column=1, sticky=W,columnspan=2)
# menu_correct_time.config(width=8)

# # Attach the trace to the StringVar, monitoring changes
# tkvar_correct_time.trace('w', on_correct_time_change)
# bt_correct_time = Button(frame_perf, text="calc",command=press_correct_time,width=bt_small,font=bt_font)
# bt_correct_time.grid(column=3, row=1,sticky=W)


# Blank line
Label(frame_perf, text= "",justify='left').grid(row=2,column=0,sticky=W,padx=(10,40))

# Quantitative perfusion
lb_quantitative = Label(frame_perf, text= "Quantitative Perfusion",justify='left')
lb_quantitative.grid(row=3,column=0,sticky=W,padx=(10,40))


# Switch data to "original conc" , gammavar fit or gammavar approx
# Since I had some bug with gammavar approx, I revoved the option to switch data to
# This will simplyfy the flow for now.  If Relative perf = Model free, aif is based on model free metrics and deconvolution based on "raw data" signal
# If Relative perf = Gamma-var afni, aif is based on the gamma var fit parameters and the deconvolution will be based on the gammavar fit signal
lb_switch_data = Label(frame_perf, text= "Using dataset",justify='left')
lb_switch_data.grid(row=4,column=0,sticky=W,padx=(10,40))

switch_data_method = StringVar()
choices = ['Original','GamVar-afni']
protocol_menu = OptionMenu(frame_perf, switch_data_method, *choices)
switch_data_method.set('Original') # set the default option
protocol_menu.grid(row=4, column=1, sticky=W, columnspan=2, padx=(0,0))
protocol_menu.config(width=8)

bt_calc_switch_data = Button(frame_perf, text="load",command=press_switch_data,width=bt_small,font=bt_font,activeforeground="red")
bt_calc_switch_data.grid(column=3, row=4,sticky=W, padx=(0,0))




# LINE 07
lb_aif = Label(frame_perf, text="Arterial Input Function")
lb_aif.grid(column=0, row=7,sticky=W,padx=(10,10))
lb_nvox = Label(frame_perf, text="N. voxels")
lb_nvox.grid(column=1, row=7,sticky=W,padx=(0,0))

txt_nvox = Text(frame_perf,width=3,height=1,font='red',relief=SUNKEN,borderwidth=2)
txt_nvox.grid(column=2, row=7,sticky=W,padx=(0,0))

bt_calc_aif = Button(frame_perf, text="calc",command=calc_aif,width=bt_small,font=bt_font,fg="black")
bt_calc_aif.grid(column=3, row=7,sticky=W, padx=(0,0),pady=(0,0))



# LINE 08
# Quantitative perfusion
lb_quantperf = Label(frame_perf, text="Deconvolution",justify='left')
lb_quantperf.grid(row=10,column=0,sticky=W,padx=(10,40))


quant_method = StringVar()
quant_choices = ['SVD','oSVD','Residue Exp']
quant_menu = OptionMenu(frame_perf, quant_method, *quant_choices)
quant_method.set('SVD') # set the default option
quant_menu.grid(row=10, column=1, sticky=W, columnspan=2, padx=(0,0))
quant_menu.config(width=8)

bt_calc_quantperf = Button(frame_perf, text="calc",command=press_calc_quantperf,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_calc_quantperf.grid(column=3, row=10,sticky=W, padx=(0,0))

# Button to show quantitative perfusion maps
bnt_view_quantperf = Button(frame_perf, text="view",command=view_quantperf,width=bt_small,font=bt_font,fg="black")
bnt_view_quantperf.grid(column=4, row=10,sticky=W, padx=(0,0))



# Show/Hide overlay 
chk_map_state = BooleanVar()
chk_map_state.set(True) #set check state
chk_map = Checkbutton(frame_chkbt, text='Overlays',command=view_overlay,var=chk_map_state)
chk_map.grid(column=0, row=1,sticky=W,padx=(10,0),pady=(10,0))

# Show/Hide raw func 
chk_func_state = BooleanVar()
chk_func_state.set(True) #set check state
chk_func = Checkbutton(frame_chkbt, text='func/anat',command=view_func,var=chk_func_state)
chk_func.grid(column=0, row=2,sticky=W,padx=(300,0),pady=(0,0),columnspan=2)

# Hide/show average brain
chk_ave_brain_state = BooleanVar()
chk_ave_brain_state.set(True) #set check state
chk_ave_brain = Checkbutton(frame_chkbt, text='Average Brain',command=view_ave_brain,var=chk_ave_brain_state)
chk_ave_brain.grid(column=0, row=1,sticky=W,padx=(150,0),pady=(10,0),columnspan=2)

# Hide/show Multiple AIF
chk_aif_state = BooleanVar()
chk_aif_state.set(True) #set check state
chk_aif = Checkbutton(frame_chkbt, text='Multiples AIFs',command=view_aif,var=chk_aif_state)
chk_aif.grid(column=0, row=2,sticky=W,padx=(10,0),pady=(0,0))

# Hide/show Avererage AIF
chk_ave_aif_state = BooleanVar()
# chk_ave_aif_state.set(False) #set check state
chk_ave_aif = Checkbutton(frame_chkbt, text='Average AIF',command=view_ave_aif,var=chk_ave_aif_state)
chk_ave_aif.grid(column=0, row=2,sticky=W,padx=(150,0),pady=(0,0),columnspan=2)


# Autoscale / Freeze scale
chk_autoscale_state = BooleanVar()
chk_autoscale_state.set(True) #set check state
chk_autoscale = Checkbutton(frame_chkbt, text='Autocsale',command=autoscale_graph,var=chk_autoscale_state)
chk_autoscale.grid(column=0, row=1,sticky=W,padx=(300,0),pady=(10,0),columnspan=2)

# Flip reference waveform
chk_flipref_state = BooleanVar()
chk_flipref_state.set(False) #set check state
chk_flipref = Checkbutton(frame_chkbt, text='Flip ref',command=flip_ref,var=chk_flipref_state)
chk_flipref.grid(column=0, row=3,sticky=W,padx=(10,0),pady=(0,0),columnspan=2)

# Show/Hide cross
chk_cross_state = BooleanVar()
chk_cross_state.set(True) #set check state
chk_cross = Checkbutton(frame_chkbt, text='Cross',command=view_cross,var=chk_cross_state)
chk_cross.grid(column=0, row=3,sticky=W,padx=(150,0),pady=(0,0),columnspan=2)

# Show/Hide Label
chk_label_state = BooleanVar()
chk_label_state.set(True) #set check state
chk_label = Checkbutton(frame_chkbt, text='Label',command=view_label,var=chk_label_state)
chk_label.grid(column=0, row=3,sticky=W,padx=(300,0),pady=(0,0),columnspan=2)

lb_regression = Label(frame_cvr, text="Linear Regression",justify='left')
lb_regression.grid(row=0,column=0,sticky=W,padx=(10,40))


type_cvr_ref = StringVar()
choice_cvr_ref = ['Stimulus ...','Enter OFF/ON','Enter ref file']
menu_cvr_ref = OptionMenu(frame_cvr, type_cvr_ref, *choice_cvr_ref)
type_cvr_ref.set('Stimulus ...') # set the default option
menu_cvr_ref.grid(row=0, column=1, sticky=W, columnspan=3, padx=(0,0))
menu_cvr_ref.config(width=8)
type_cvr_ref.trace('w', calc_cvr_ref)

bt_shift_left = Button(frame_cvr, text="<<",command=press_shift_cvr_ref_left,width=bt_small,font=bt_font,activeforeground="red")
bt_shift_left.grid(column=4, row=0,sticky=W)

bt_shift_right = Button(frame_cvr, text=">>",command=press_shift_cvr_ref_right,width=bt_small,font=bt_font,padx=1,activeforeground="red")
bt_shift_right.grid(column=5, row=0,sticky=W)

bt_calc_regression = Button(frame_cvr, text="calc",command=press_calc_regression,width=bt_small,font=bt_font,fg="black")
bt_calc_regression.grid(column=6, row=0,sticky=W)
bt_view_regression = Button(frame_cvr, text="view",command=press_view_regression,width=bt_small,font=bt_font,fg="black")
bt_view_regression.grid(column=7, row=0,sticky=W)


# LINE 07

# Lag analysis
lb_lag = Label(frame_cvr, text="LAG analysis",justify='left')
lb_lag.grid(column=0,row=1,sticky=W,padx=(10,40))

box_lag_minmax = Text(frame_cvr,height=1,width=time_map_width,relief=SUNKEN,borderwidth=2)
box_lag_minmax.insert('1.0', 'ST=0 ET=8 DT=0.2')  # Insert default text at the start (line 1, character 0)
box_lag_minmax.grid(column=1, row=1,sticky=W,columnspan=4)

bt_calc_lag_2d = Button(frame_cvr, text="👁",command=press_calc_lag_2d,width=bt_small,font=bt_font,fg="black")
bt_calc_lag_2d.grid(column=5, row=1,sticky=W)

bt_calc_lag = Button(frame_cvr, text="calc",command=press_calc_lag_3d,width=bt_small,font=bt_font,fg="black")
bt_calc_lag.grid(column=6, row=1,sticky=W)

bt_view_lag = Button(frame_cvr, text="view",command=press_view_lag,width=bt_small,font=bt_font,fg="black")
bt_view_lag.grid(column=7, row=1,sticky=W)

# Tau analysis

lb_tau = Label(frame_cvr, text="RT   analysis",justify='left')
lb_tau.grid(column=0,row=2,sticky=W,padx=(10,40))

box_tau_minmax = Text(frame_cvr,height=1,width=time_map_width,relief=SUNKEN,borderwidth=2)
box_tau_minmax.insert('1.0', 'ST=0 ET=8 DT=0.2')  # Insert default text at the start (line 1, character 0)
box_tau_minmax.grid(column=1, row=2,sticky=W,columnspan=4)

bt_calc_tau_2d = Button(frame_cvr, text="👁",command=press_calc_tau_2d,width=bt_small,font=bt_font,fg="black")
bt_calc_tau_2d.grid(column=5, row=2,sticky=W)

bt_calc_tau = Button(frame_cvr, text="calc",command=press_calc_tau_3d,width=bt_small,font=bt_font,fg="black")
bt_calc_tau.grid(column=6, row=2,sticky=W)

bt_view_tau = Button(frame_cvr, text="view",command=press_view_tau,width=bt_small,font=bt_font,fg="black")
bt_view_tau.grid(column=7, row=2,sticky=W)


# Define Time windows
lb_time_window = Label(frame_cvr, text="Ave. Time-windows")
lb_time_window.grid(column=0, row=3,sticky=W,padx=(10,10))

type_bands = StringVar()
choice_bands = ['Define...','Inputs','Auto']
menu_bands = OptionMenu(frame_cvr, type_bands, *choice_bands)
type_bands.set('Define...') # set the default option
menu_bands.grid(row=3, column=1, sticky=W, columnspan=2)
menu_bands.config(width=4)
type_bands.trace('w', press_define_windows)

bt_move_bands_left = Button(frame_cvr, text="<",command=move_bands_left,width=bt_small,font=bt_font,fg="black")
bt_move_bands_left.grid(column=3, row=3,sticky=W)

bt_move_bands_right = Button(frame_cvr, text=">",command=move_bands_right,width=bt_small,font=bt_font,fg="black")
bt_move_bands_right.grid(column=4, row=3,sticky=W)

bt_shrink_bands = Button(frame_cvr, text="><",command=shrink_bands,width=bt_small,font=bt_font,fg="black")
bt_shrink_bands.grid(column=5, row=3,sticky=W)

bt_dilate_bands = Button(frame_cvr, text="<>",command=dilate_bands,width=bt_small,font=bt_font,fg="black")
bt_dilate_bands.grid(column=6, row=3,sticky=W)

bt_ave_time = Button(frame_cvr, text="calc",command=press_ave_boluses,width=bt_small,font=bt_font,fg="black")
bt_ave_time.grid(column=7, row=3,sticky=W)





# LINE 07
lb2_aif = Label(frame_cvr, text="Arterial Input Function")
lb2_aif.grid(column=0, row=9,sticky=W,padx=(10,10))

lb2_nvox = Label(frame_cvr, text="N. voxels")
lb2_nvox.grid(column=1, row=9,sticky=W)

txt2_nvox = Text(frame_cvr,width=3,height=1,font='red',relief=SUNKEN,borderwidth=2)
txt2_nvox.grid(column=2, row=9,sticky=W)

type_aif = StringVar()
choice_aif = ['Pos','Neg']
menu_cvr_aif = OptionMenu(frame_cvr, type_aif, *choice_aif)
type_aif.set('Neg') # set the default option
menu_cvr_aif.grid(row=9, column=3, sticky=W, columnspan=2)
#type_aif.trace('w', calc_aif_cvr)

bt2_calc_aif = Button(frame_cvr, text="calc",command=press_calc_aif_cvr,width=bt_small,font=bt_font,fg="black")
bt2_calc_aif.grid(column=5, row=9,sticky=W)


lb_quantperf = Label(frame_cvr, text="Quantitative Perfusion",justify='left')
lb_quantperf.grid(row=18,column=0,sticky=W,padx=(10,40))

quant_method_cvr = StringVar()
quant_choices_cvr = ['SVD','oSVD']
quant_menu_cvr = OptionMenu(frame_cvr, quant_method_cvr, *quant_choices_cvr)
quant_method_cvr.set('SVD') # set the default option
quant_menu_cvr.grid(row=18, column=1, sticky=W, columnspan=3)
quant_menu_cvr.config(width=8)

# Button to calc quantitative perfusion maps
bt_calc_quantperf = Button(frame_cvr, text="calc",command=calc_quantcvr,width=bt_small,font=bt_font,fg="black",activeforeground="red")
bt_calc_quantperf.grid(column=4, row=18,sticky=W)

# Button to show quantitative perfusion maps
bnt_view_quantperf = Button(frame_cvr, text="view",command=view_quantperf,width=bt_small,font=bt_font,fg="black")
bnt_view_quantperf.grid(column=5, row=18,sticky=W)


# Make matplotlib figure called fig with 3x2 axes.
# ====================================================================================================

fig = plt.figure(figsize=(14,9))
gs = GridSpec(2, 3, figure=fig)

# Create the subplots using GridSpec
ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
ax3 = fig.add_subplot(gs[0,1:3])  # First row, merge second and third columns (spans two columns)
ax4 = fig.add_subplot(gs[1, 0])  # Second row, first column
ax5 = fig.add_subplot(gs[1, 1])  # Second row, second column
ax6 = fig.add_subplot(gs[1, 2])  # Second row, third column

for ax in [ax1, ax3, ax4, ax5, ax6]:
    ax.set_facecolor('black')


fig.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.05,right=0.95)

# Revove axes and tick for all subplots
for ax in [ax1, ax5 ,ax6 ,ax4]:
    ax.set_xticks([])
    ax.set_yticks([])

ax3.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax3.xaxis.set_tick_params(labelcolor='white',color='none',pad=-15)
ax3.yaxis.set_tick_params(labelcolor='white',color='none',pad=-20)
text_coord = ax3.text(0.6, 0.95, '', fontsize=10, color='white', ha='left', va='top', transform=ax3.transAxes)
for spine in ax3.spines.values():
    spine.set_visible(False)  # Completely hide the spines



# Create a style and get background color from it
style = ttk.Style()
bg_color = style.lookup('TFrame', 'background')

# Fallback if lookup fails (esp. on some Windows themes)
if not bg_color or bg_color in ('', 'SystemWindow'):
    if platform.system() == 'Darwin':
        bg_color = 'systemWindowBackgroundColor'  # Still valid on macOS
    elif platform.system() == 'Windows':
        bg_color = window.cget('bg')  # Default window background

# Remove background color
fig.patch.set_alpha(0.0)

# Place fig in Tk window.
frame_fig_container = Frame(window)
frame_fig_container.pack(side=RIGHT, fill=BOTH, expand=True)
#progress = ttk.Progressbar(frame_fig_container, orient="vertical",length=300, mode="indeterminate")
#progress.pack(side="left", anchor="nw")
canvas_fig = FigureCanvasTkAgg(fig, master=frame_fig_container)
canvas_fig.draw()
canvas_fig.get_tk_widget().pack(fill=BOTH,expand=True)
canvas_fig.get_tk_widget().config(bg=bg_color)


# Create the Horizontal RangeSlider axis at the bottom
xlim_slider_ax = fig.add_axes([0.45, 0.97, 0.4, 0.01], facecolor='lightgoldenrodyellow')
xlim_slider = RangeSlider(xlim_slider_ax, '', -2, 2, valinit=(-1, 1), orientation='horizontal',      
                     track_color='lightgrey',
                     handle_style = {'facecolor':'white','edgecolor': 'white','size':'8'})    # Set the color of the track

xlim_slider.valtext.set_visible(False)  # This will hide the handle value text
xlim_slider.closedmax = False  

# Create the Vertical RangeSlider axis on the right side
ylim_slider_ax = fig.add_axes([0.95, 0.55, 0.01, 0.4], facecolor='lightgoldenrodyellow')
ylim_slider = RangeSlider(ylim_slider_ax, '', -2, 2, valinit=(-1, 1), orientation='vertical',      
                     track_color='lightgrey',
                     handle_style = {'facecolor':'white','edgecolor': 'white','size':'8'})    # Set the color of the track

ylim_slider.valtext.set_visible(False)  # This will hide the handle value text
ylim_slider.closedmax = False  







# Under construction
def roll_view(event):
    global anat
    global data4,data5,data6
    global und4, und5, und6
    global ove4, ove5, ove6

    anat = anat.transpose(2,0,1)
    und4.set_data(anat[:,:, j0])
    und5.set_data(anat[:,:, j0])
    und6.set_data(anat[:,:, j0])


    data4 = data4.transpose(2,0,1)
    ove4.set_data(data4[...,J0])

    data5 = data5.transpose(2,0,1)
    ove5.set_data(data5[...,J0])
    
    data6 = data6.transpose(2,0,1)
    ove6.set_data(data6[...,J0])
    
    
    
    plt.draw()


# This list store the lines and the associated voxel coordinates
ijk_lines = []

# Connect some functions with some button events
ylim_slider.on_changed(update_ylim)
xlim_slider.on_changed(update_xlim)
fig.canvas.mpl_connect('motion_notify_event', reset_slider_limits)
fig.canvas.mpl_connect('motion_notify_event',on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_click_im_right)
fig.canvas.mpl_connect('pick_event', on_pick_line)
fig.canvas.mpl_connect('button_press_event', on_click_time)
fig.canvas.mpl_connect('motion_notify_event', on_move_time)
fig.canvas.mpl_connect('button_release_event', on_release_time)
cid_on_click_im = fig.canvas.mpl_connect('button_press_event', on_click_im)

vline, text, cid = on_move_time_set(ax3)

# Create some menus to choose color scales
context_menu4 = Menu(window, tearoff=0)
context_menu5 = Menu(window, tearoff=0)
context_menu6 = Menu(window, tearoff=0)

# Define ROI 6 colors
roi_colors = [
    'red',         
    'green',        
    'blue',         
    'orange',      
    'purple',       
    '#b58900'       # darker yellow (hex code)
]

# Create a ListedColormap
cmap_roi = ListedColormap(roi_colors, name='roi_colors')

# Define BH, CVR, fMRI colors scale
colors = [
    (0.0, "#000099"), 
    (0.1, "#0000ff"), 
    (0.49995, "#98f5ff"),
    (0.5, "#ffffff"),
    (0.50005, "#ffff00"), 
    (0.75, "#ff0000"), 
    (0.95, "#5B004E"),    
    (1.0, "#5B004E"),          
]

# Extract the positions and corresponding hex colors separately
positions, hex_colors = zip(*colors)

# Create the custom colormap
fMRI_colors = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, hex_colors)))

# Define your custom "cool" part
colors = [
    (0, "#98f5ff"),
    (0.5, "#0000ff"),
    (0.8, "#000099"), 
    (1, "#000000"),
]
positions, hex_colors = zip(*colors)
cool_cmap = LinearSegmentedColormap.from_list("custom_cool", list(zip(positions, hex_colors)))

# Sample colors from your custom "cool" colormap
n = 256
cool_colors = cool_cmap(np.linspace(0, 1, n // 2))
hot_colors  = plt.cm.hot(np.linspace(0, 1, n // 2))

# Combine (cool for [0–0.5], hot for [0.5–1])
combined_colors = np.vstack((cool_colors, hot_colors))
CoolHot_colors = ListedColormap(combined_colors, name='CoolHot')




# Define SEGMENTATION colors
colors = [
    (0/30, "#00ff00"), 
    (1/30, "#00ff00"), 
    (2/30, "#FF0000"), 
    (3/30, "#0000ff"),
    (10/30, "#FBFF00"),
    (20/30, "#F900E0"), 
    (30/30, "#03E2FF")     
]

# Extract the positions and corresponding hex colors separately
positions, hex_colors = zip(*colors)

# Create the custom colormap
Seg_colors = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, hex_colors)))

# Create a dictionary for colormap names and colormap objects
colormaps = {
    "hot": plt.cm.hot,
    "viridis": plt.cm.viridis,
    "plasma": plt.cm.plasma,
    "inferno": plt.cm.inferno,
    "bone": plt.cm.bone,
    "turbo": plt.cm.turbo,
    "CoolHot":CoolHot_colors,
    "fMRI": fMRI_colors,  # Add custom colormap with a string name
    "Seg": Seg_colors # Add the SEG colorscale
}


# Populate the context menus with the colormap names and objects
for cmap_name, cmap_object in colormaps.items():
    context_menu4.add_command(label=cmap_name, command=partial(on_menu_selection4, cmap_object))
    context_menu5.add_command(label=cmap_name, command=partial(on_menu_selection5, cmap_object))
    context_menu6.add_command(label=cmap_name, command=partial(on_menu_selection6, cmap_object))

interface_vars = [key for key in globals() if not key.startswith("__")]

window.mainloop()



# Notes on masks that I never remember (applying masked)

# # Apply the mask from MA to B
# MB = np.ma.masked_where(MA.mask, B)
# MB = np.ma.masked_array(B, mask=MA.mask)

# mask = np.ma.getmask(MA)
# MB = np.ma.masked_array(B, mask=mask)

# MB = B.copy()
# MB[MA.mask] = np.ma.masked  # Mask elements in B where MA is masked

# MB = np.ma.masked_where(MA > threshold, B)