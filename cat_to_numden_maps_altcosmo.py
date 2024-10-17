# Takes theta,phi lists for clusters and galaxies, plots them, and smooths them

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import z_at_value
import astropy.units as u
import coop_setup_funcs as csf
import websky as ws
from colossus.cosmology import cosmology
P18 = cosmology.setCosmology('planck18-only') # consistent with the Peak Patch setup
P18_astropy = P18.toAstropy()
import h5py as h5
import sys

# take in arguments for Omega_M, Omega_b, h, optional add-on string
if len(sys.argv)<5:
    sys.exit("Need to specify Omega_M, Omega_b, h, OMstring")
omegam = float(sys.argv[1])
omegab = float(sys.argv[2])
h = float(sys.argv[3])
OMstring = sys.argv[4]
    
# inputs for different options
split  = False
masswgt_odmap = True

#45 * u.Mpc
if split:
    maptype = 'ng' # or 'nu'
obj = 'halos'
    
if "_const_comov" in OMstring:
    constant_comoving = True # if True, use single comoving distance range for all cosmologies rather than a fixed redshift range
else:
    constant_comoving = False
    
# smth_scale    = 8 / P18.h * u.Mpc # kind of like sigma 8
smth_scale    = 10 * u.Mpc 
smthmode = 'gaussian'

new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam, Ob0=omegab, H0=h*100.)
new_cosmo_astropy = new_cosmo.toAstropy()
if constant_comoving:
    cent = new_cosmo_astropy.comoving_distance(0.5)
    # find the redshift range that encompasses 200 Mpc in this cosmology, centered at z=0.5
    zbin = [z_at_value(new_cosmo_astropy.comoving_distance, cent-100*u.Mpc), z_at_value(new_cosmo_astropy.comoving_distance, cent+100*u.Mpc)]
else: # get the redshift range for a 200 Mpc-wide bin in the fiducial cosmology, apply that redshift range to all cosmologies
    zbin = [z_at_value(P18_astropy.comoving_distance, P18_astropy.comoving_distance(0.5)-100*u.Mpc), z_at_value(P18_astropy.comoving_distance, P18_astropy.comoving_distance(0.5)+100*u.Mpc)]


if (smth_scale == 0*u.Mpc) or (smth_scale==0):
    smth_str = ''
else:
    smth_str = '_smth_'+str(int(smth_scale.value))+'Mpc'

print("Omega M = {:.2f}".format(omegam))
mask_path = None
# "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_lowright.fits"
if obj=='halos':
    outpath  = "/mnt/raid-cita/mlokken/pkpatch/number_density_maps/alt_cosmo/"
    pkscfile = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.pksc.13579".format(OMstring.replace("_const_comov",""))
    npfile   = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.npy".format(OMstring.replace("_const_comov",""))
    hdf5file = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.hdf5".format(OMstring.replace("_const_comov",""))
    # if not os.path.exists(npfile):
    # ws.pksc_to_npy(pkscfile, new_cosmo, min_mass=10**13) # create the .npy version
    ##  this is really slow!!
    # if not os.path.exists(hdf5file):
    #     ws.pksc_to_hdf5(pkscfile, new_cosmo, min_mass=10**13) # create the .hdf5 version
    
if obj=='galaxies':
    npfile = "/mnt/scratch-lustre/mlokken/pkpatch/galaxy_catalogue.h5"
    outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/"
min_mass = 10**13 # to be cnosistent for all cosmologies, as they have different lower mass limits
max_mass = 10**17 # include everything on the high end
if min_mass==None and max_mass==None:
    mass_str = 'all_'
else:
    mass_str = '{:.0e}_{:.0e}_'.format(min_mass,max_mass)
print("Reading catalog.")
if obj=='halos':
    halofile = h5.File(hdf5file, 'r')
    ra, dec, z, mass = halofile['ra'][:], halofile['dec'][:], halofile['z'][:], halofile['m200m'][:]
    inmass = (mass>min_mass) & (mass<max_mass)
    ra, dec, z, mass = ra[inmass], dec[inmass], z[inmass], mass[inmass]
    halofile.close()
    # ra, dec, z, chi, mass = ws.read_halos(npfile, min_mass, max_mass)
elif obj=='galaxies':
    ra, dec, z, chi, mass = ws.read_galcat(npfile, min_mass, max_mass, satfrac=.15)
catlen = len(ra)
print("Catalog read.")
if masswgt_odmap:
    w = mass/(10**12)
    
else:
    w = 1


# a redshift bin surrounding z=0.5 that is 200 Mpc in the Planck 2018 cosmology    
nside = 4096

if mask_path is not None:
    mask = hp.read_map(mask_path)
else:
    mask = None

bin = (z<zbin[1]) & (z>zbin[0])
theta,phi = csf.DeclRatoThetaPhi(dec[bin],ra[bin])

print(np.amin(theta),np.amax(theta), np.amin(phi), np.amax(phi))
thetaphi = np.zeros((len(theta),2))
thetaphi[:,0]=theta
thetaphi[:,1]=phi
weight = w[bin]
z_mid = ((zbin[0]+zbin[1])/2.).value
print("Middle of slice redshift ", z_mid)
# get the redshift at the middle of this slice
print("Getting overdensity and number density maps for bin ", zbin)

if smth_scale > 0:
    # get the angular size (function gives arcseconds per kpc, convert to
    # Mpc, then multiply by user-input scale [in Mpc]
    if constant_comoving:
        smth_scale_arcsec = new_cosmo_astropy.arcsec_per_kpc_comoving(z_mid).to(u.arcsec/u.megaparsec)*smth_scale # constant comoving smoothing scale
    else:
        smth_scale_arcsec = P18_astropy.arcsec_per_kpc_comoving(z_mid).to(u.arcsec/u.megaparsec)*smth_scale # constant angular smoothing scale for all cosmologies
    smth_str += '_'+str(int(smth_scale_arcsec.value//60.))+'a'
else:
    smth_scale_arcsec = 0*u.arcsec
label = 'ndmap'

outfile = "{:s}_{:s}_100_{:s}z_{:s}_{:s}{:s}_{:s}.fits".format(OMstring, label, mass_str, str(zbin[0].value).replace('.','pt')[:6], str(zbin[1].value).replace('.','pt')[:6], smth_str, smthmode)
if os.path.exists(os.path.join(outpath,outfile)):
    print("Map already made. Moving on.\n")
else:
    if smthmode == 'gaussian':
        map = csf.get_nd_map(nside, thetaphi[:,0], thetaphi[:,1], mask, smth_scale_arcsec.value, wgt=weight)
    elif smthmode == 'tophat':
        map = csf.get_od_map(nside, thetaphi[:,0], thetaphi[:,1], mask, smth_scale_arcsec.value, wgt=weight, beam='tophat')
    print("Writing map to %s" %outpath+outfile)
    hp.write_map(outpath+outfile, map, overwrite=True)

