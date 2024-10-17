import numpy as np
import error_analysis_funcs as ef
import os
import astropy.units as u
import subprocess
import coop_setup_funcs as csf
from colossus.cosmology import cosmology
# set standard cosmology (Planck 18)
P18 = cosmology.setCosmology('planck18-only') # consistent with the Peak Patch setup
P18_omegam = P18.Om0
P18_astropy = P18.toAstropy()
from astropy.cosmology import z_at_value
import h5py as h5
import healpy as hp
import sys

    
def unmasked_idx(ra, dec, mask):
    mask = hp.read_map(mask)
    bool_mask = np.full(len(ra), False)
    map_pix = hp.ang2pix(hp.get_nside(mask), ra, dec, lonlat=True)
    for i in range(len(ra)):
        if mask[map_pix[i]]==1:
            bool_mask[i] = True
    return bool_mask

# take in arguments for Omega_M, Omega_b, h, optional add-on string
if len(sys.argv)<5:
    sys.exit("Need to specify Omega_M, Omega_b, h, OMstring.")
omegam = float(sys.argv[1])
omegab = float(sys.argv[2])
h = float(sys.argv[3])
OMstring = sys.argv[4]
    

nu_e_cuts = True
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
smth_arcmin     = 0*u.arcmin
smth_scale    = 20 * u.Mpc 
if (smth_scale == 0*u.Mpc) or (smth_scale==0):
    smth_str = ''
else:
    smth_str = 'smth_'+str(int(smth_scale.value))+'Mpc'
    
if "_const_comov" in OMstring:
    constant_comoving = True # if True, use single comoving distance range for all cosmologies rather than a fixed redshift range
else:
    constant_comoving = False
orient_mode = 'halos'
# split if you want to only use some of the galaxy data to orient and other to stack
split = False

if split:
    pct = 75 # only use 75 percent of the galaxy data for orientation, as the other 25 percent is stacked
else:
    pct = 100

# cut the clusters or halos that will be stacked by richness or mass

cut = 'mass'
cutmin = 5*10**13
cutmax = None

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
pkmap_path  = "/mnt/raid-cita/mlokken/pkpatch/number_density_maps/alt_cosmo/"
clmask      = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup_reduced5.fits"
pkmask      = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup.fits"
ymask       = "/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup.fits"
outpath     = "/mnt/raid-cita/mlokken/pkpatch/alt_cosmo_stacks/"
peakspath = outpath + "orient_by_{:s}_{:d}".format(orient_mode, pct)
if not os.path.exists(peakspath):
    os.mkdir(peakspath)


new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam, Ob0=omegab, H0=h*100.)
new_cosmo_astropy = new_cosmo.toAstropy()

# get the galaxy and cluster bins in redshift / comoving distance
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


hdf5file = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.hdf5".format(OMstring.replace("_const_comov",""))
halofile = h5.File(hdf5file, 'r')
ra, dec, z, mass = halofile['ra'][:], halofile['dec'][:], halofile['z'][:], halofile['m200m'][:]
# get halos in the right mass range for stacking
inmass = (mass>cutmin)
ra, dec, z, mass = ra[inmass], dec[inmass], z[inmass], mass[inmass]
print("Number of halos in mass range: ", len(ra))
halofile.close()
# cut right at the beginning to save time and memory
inz = (zcent-0.1<z)&(z<zcent+0.1)
ra,dec,z,mass = ra[inz], dec[inz], z[inz], mass[inz]
pkmap = pkmap_path + f"{OMstring}_odmap_100_1e+13_1e+17_z_{str(zbin[0].value).replace('.','pt')[:6]}_{str(zbin[1].value).replace('.','pt')[:6]}_{smth_str}_gaussian.fits".format(OMstring)
print("Using {:s} as the peak map.".format(pkmap))

bincent = (cl_dlow + cl_dhi)/2.
print("Finding clusters within 50 cMpc (standard cosmo) of {:.0f} Mpc".format(bincent))
print("In redshift space, this is between {:.2f} and {:.2f}.".format(cl_zlow,cl_zhi)) # this should be the same for all cosmos
cl_inbin    = (cl_zlow<z)&(z<cl_zhi)
binstr_cl= str(cl_zlow).replace('.','pt')[:6]+"_"+str(cl_zhi).replace('.','pt')[:6]
binstr_orient = str(zbin[0]).replace('.','pt')[:6]+"_"+str(zbin[1]).replace('.','pt')[:6]

# reduce clusters to those that are not masked
unmasked = unmasked_idx(ra, dec, clmask)
theta,phi = csf.DeclRatoThetaPhi(dec[unmasked],ra[unmasked])
thetaphi_bin = np.array([theta, phi]).T
tp_file     = os.path.join(peakspath, "thetaphi_{:s}_{:s}.txt".format(binstr_cl, cutstr))
np.savetxt(tp_file, thetaphi_bin)

# set up the run
pkmap = os.path.join(pkmap_path, pkmap)
# make the ini files
inifile_root = "pkpatch_{:s}_{:s}_{:s}_{:s}{:s}_orient_{:d}pct_{:s}_{:s}".format(OMstring, cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
print(peakspath + "/" + inifile_root+"_pks.fits")
if not os.path.exists(os.path.join(peakspath, inifile_root+"_pks.fits")):
    print("Orienting by surrounding galaxies.\n")
    pk_ini = ef.make_pk_ini_file_angular(pkmap, standard_pk_file, peakspath, inifile_root, smth_arcmin, thetaphi_file=tp_file, pk_mask=pkmask, e_min=e_min, e_max=e_max, nu_min=nu_min)
    print("Running GetPeaks on {:s}".format(pk_ini))
    subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
else:
    print("Already run. Moving on.")
