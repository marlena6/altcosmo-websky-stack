import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import z_at_value
import astropy.units as u
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
from astropy.io import fits
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
# set standard cosmology (Planck 18)
P18 = cosmology.setCosmology('planck18-only') # consistent with the Peak Patch setup
P18_astropy = P18.toAstropy()
P18_omegam = P18.Om0
import h5py as h5
import healpy as hp
import sys


# take in arguments for Omega_M, Omega_b, h, optional add-on string
if len(sys.argv)<5:
    sys.exit("Need to specify Omega_M, Omega_b, h, OMstring.")
omegam = float(sys.argv[1])
omegab = float(sys.argv[2])
h = float(sys.argv[3])
OMstring = sys.argv[4]

if "_const_comov" in OMstring:
    constant_comoving = True # if True, use single comoving distance range for all cosmologies rather than a fixed redshift range
else:
    constant_comoving = False
# ALL CHOICES COME BELOW
########################################################
nu_e_cuts = True
# Input here which maps to stack
stack_y        = True
# Smooth the maps by a Gaussian with this beam FWHM
smth     = 20 #Mpc
# split if you want to only use some of the galaxy data to orient and other to stack
split = False
# cut the clusters or halos that will be stacked by richness or mass

cut = 'mass'
cutmin = 5*10**13
cutmax = None
########################################################


if nu_e_cuts:
    pt_selection_str = "nugt2_egtpt3_"
    e_min  = 0.3
    e_max  = None
    nu_min = 2
else:
    pt_selection_str = ''
    e_min = None
    e_max = None
    nu_min = None
    
zcent = 0.5
# Smooth the maps by a Gaussian with this beam FWHM
orient_mode = 'halos'

if split:
    pct = 75 # only use 75 percent of the galaxy data for orientation, as the other 25 percent is stacked
else:
    pct = 100
    
smth_arcmin     = 0*u.arcmin
smth_scale    = 20 * u.Mpc 
if (smth_scale == 0*u.Mpc) or (smth_scale==0):
    smth_str = ''
else:
    smth_str = 'smth_'+str(int(smth_scale.value))+'Mpc'
    

if cutmin is not None:
    cutminstr = 'gt{:.0e}'.format(cutmin)
else:
    cutminstr = ''
if cutmax is not None:
    cutmaxstr = 'lt{:.0e}'.format(cutmax)
else:
    cutmaxstr = ''

cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr

standard_stk_file = "/home/mlokken/oriented_stacking/general_code/standard_stackfile.ini"
standard_pk_file  = "/home/mlokken/oriented_stacking/general_code/standard_pkfile.ini"
ymap_path  = "/mnt/raid-cita/mlokken/pkpatch/ymaps/"
clmask      = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup_reduced5.fits"
pkmask      = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup.fits"
ymask       = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup.fits"
outpath     = "/mnt/raid-cita/mlokken/pkpatch/alt_cosmo_stacks/"
stkpath = outpath + "orient_by_{:s}_{:d}/stacks".format(orient_mode, pct)
if not os.path.exists(stkpath):
    os.mkdir(stkpath)


ymap = ymap_path + "2400Mpc_n442_nb28_nt4_{:s}_merge_battaglia_car_1p6arcmin_cutoff4_4096_hpx.fits".format(OMstring.replace("_const_comov",""))

new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam, Ob0=omegab, H0=h*100.)
new_cosmo_astropy = new_cosmo.toAstropy()

if constant_comoving:     # find the redshift range that encompasses 200 Mpc in this cosmology, centered at z=0.5
    cent = new_cosmo_astropy.comoving_distance(0.5)
    zbin = [z_at_value(new_cosmo_astropy.comoving_distance, cent-100*u.Mpc), z_at_value(new_cosmo_astropy.comoving_distance, cent+100*u.Mpc)]
    cl_dlow  = new_cosmo_astropy.comoving_distance(zcent).value-50
    cl_dhi   = new_cosmo_astropy.comoving_distance(zcent).value+50
    cl_zlow  = z_at_value(new_cosmo_astropy.comoving_distance, cl_dlow*u.Mpc).value
    cl_zhi   = z_at_value(new_cosmo_astropy.comoving_distance, cl_dhi*u.Mpc).value
    
else: # get the redshift range for a 200 Mpc-wide bin in the fiducial cosmology, apply that redshift range to all cosmologies
    zbin = [z_at_value(P18_astropy.comoving_distance, P18_astropy.comoving_distance(0.5)-100*u.Mpc), z_at_value(P18_astropy.comoving_distance, P18_astropy.comoving_distance(0.5)+100*u.Mpc)]
    cl_dlow  = P18_astropy.comoving_distance(zcent).value-50
    cl_dhi   = P18_astropy.comoving_distance(zcent).value+50
    cl_zlow  = z_at_value(P18_astropy.comoving_distance, cl_dlow*u.Mpc).value
    cl_zhi   = z_at_value(P18_astropy.comoving_distance, cl_dhi*u.Mpc).value


z_mid = ((zbin[0]+zbin[1])/2.).value
print("Middle of slice redshift ", z_mid)
# get the smoothing for this cosmology
if smth_scale > 0:
    if constant_comoving:
        smth_scale_arcsec = new_cosmo_astropy.arcsec_per_kpc_comoving(z_mid).to(u.arcsec/u.megaparsec)*smth_scale # constant comoving smoothing scale
    else:
        smth_scale_arcsec = P18_astropy.arcsec_per_kpc_comoving(z_mid).to(u.arcsec/u.megaparsec)*smth_scale # constant angular smoothing scale for all cosmologies
    smth_str += '_'+str(int(smth_scale_arcsec.value//60.))+'a'
else:
    smth_scale_arcsec = 0*u.arcsec
    
binstr_cl= str(cl_zlow).replace('.','pt')[:6]+"_"+ str(cl_zhi).replace('.','pt')[:6]
binstr_orient = str(zbin[0]).replace('.','pt')[:6]+"_"+ str(zbin[1]).replace('.','pt')[:6]
inifile_root = "pkpatch_{:s}_{:s}_{:s}_{:s}{:s}_orient_{:d}pct_{:s}_{:s}".format(OMstring, cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
pksfile = os.path.join(outpath+"orient_by_{:s}_{:d}/".format(orient_mode, pct), inifile_root+"_pks.fits")
y_inifile_root = "ymap"+"_"+inifile_root
# start = time.time()
stk_ini = ef.make_stk_ini_file_angular(pksfile, ymap, standard_stk_file, stkpath, y_inifile_root, stk_mask=ymask)
print("Running Stack on {:s}".format(stk_ini))
if not os.path.exists(os.path.join(stkpath, y_inifile_root+"_stk.fits")):
    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
    # remove extraneous files                                                                                                                                     
    os.remove(os.path.join(stkpath, y_inifile_root+"_stk.txt"))
    os.remove(os.path.join(stkpath, y_inifile_root+"_stk.patch"))
