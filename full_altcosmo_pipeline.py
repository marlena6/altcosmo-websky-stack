import numpy as np
import os
from colossus.cosmology import cosmology
import subprocess

# set standard cosmology (Planck 18)
P18 = cosmology.setCosmology('planck18-only') # consistent with the Peak Patch setup
P18_omegam = P18.Om0
P18_omegab = P18.Ob0
omegak = P18.Ok0
print("Fiducial omega_M:", P18_omegam)
# set values of Omega_M
omegam_varied = np.linspace(P18_omegam-0.1,P18_omegam+0.1,5)
print("Omega_M will be:", omegam_varied)

# Additional parameters
P18_h  = P18.h # dimensionless Hubble const. "little h", H_0/100 km/s/Mpc
ns     = P18.ns # spectral index
rho_c  = P18.rho_c(0)*10**9 #2.7754e11 # critical energy density in units of h^2 M_sol Mpc^-3
rho    = rho_c*P18_omegam*P18_h**2 # average matter-like energy density at present
z      = [0.5] # redshift at the present and at the redshift we will evaluate
As     = 2.100e-9 # primordial super-horizon comoving curvature power spectrum amplitude
tau    = 0.0544 # optical depth of reionization (not in colossus)
m_nu   = 0.06 # massive neutrino mass
sigma8 = P18.sigma(R=8, z=z[0])
print("Fiducial sigma 8 at z={:.2f}: ".format(z[0]), sigma8)

fix_density = False
constant_comoving = True
addon = ''
if fix_density:
    addon += '_fix_physdens'
    # don't run Omegam=0.32 because it is the same as fix_density=False version
    # omegam_varied = np.delete(omegam_varied, 2)
    omh2 = P18_omegam*P18_h**2
    obh2 = P18_omegab*P18_h**2
    omegab_varied = omegam_varied*P18_omegab/P18_omegam # fix the baryon fraction
    print("Planck18 Omega m h^2: ", omh2)
    print("Planck18 Omega b h^2: ", obh2)
    h_varied = np.sqrt(omh2/omegam_varied)
    omegac_varied = omegam_varied-omegab_varied
    print("Omega_cdm", omegac_varied)
    H0_varied = 100*h_varied
    print("H0 varied: ", h_varied)
    
else:
    omegac_varied = omegam_varied-P18_omegab
    print("Omega_b/Omega_m is:", P18_omegab/omegam_varied)
    h_varied = np.full(len(omegam_varied), P18_h) # fix the hubble constant
    H0_varied = 100*h_varied
    omegab_varied = np.full(len(omegam_varied), P18_omegab) # fix the baryon fraction

testmode = False
if testmode:
    np.asarray([np.linspace(P18_omegam-0.1,P18_omegam+0.1,5)[2]]) # only do the fiducial one
    
mask = 'lowright'
# first make the mask if it does not exist
if not os.path.exists("/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_lowright.fits"):
    print("making mask")
    ("python altcosmo_mask.py")
    subprocess.run(["python", "altcosmo_mask.py"])
'''
# make the hdf5 files if they do not exist
for i in range(len(omegam_varied)):
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], H0=H0_varied[i], Ob0=omegab_varied[i]) 
    OMstring = "OM{:d}{:s}".format(int(round(omegam_varied[i]*100)), addon)
    hdf5name = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.hdf5".format(OMstring)
    if not os.path.exists(hdf5name):
        print("making hdf5 file for {:s}".format(OMstring))
        subprocess.run(["python", "/home/mlokken/response_functions/pksc2hdf5.py", "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.pksc.13579".format(OMstring), hdf5name, str(new_cosmo.Om0), str(new_cosmo.Ob0), str(new_cosmo.h)])

# make the ymap if it does not exist
for i in range(len(omegam_varied)):
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], Ob0=omegab_varied[i], H0=H0_varied[i]) 
    OMstring = "OM{:d}{:s}".format(int(round(omegam_varied[i]*100)), addon)
    hdf5file = "/mnt/raid-cita/mlokken/pkpatch/2400Mpc_n442_nb28_nt4_{:s}_merge.hdf5".format(OMstring)
    hdf5name = hdf5file.split("/")[-1][:-5]
    ymap_out = f"/mnt/raid-cita/mlokken/pkpatch/ymaps/{hdf5name}_battaglia_car_1p6arcmin_cutoff4.fits"
    if not os.path.exists(ymap_out):        
        os.environ["JULIA_NUM_THREADS"] = "16"
        args = "--halofile {:s} --OMstring {:s} --Omega_c {:.5f} --Omega_b {:.5f} --h {:.5f}".format(hdf5file, OMstring, new_cosmo.Om0-new_cosmo.Ob0, new_cosmo.Ob0, new_cosmo.h)
        print(args)
        os.system("julia ymaps.jl {:s}".format(args))

    if not os.path.exists(ymap_out[:-5]+"_4096_hpx.fits"):
        # 
        os.system(f"python /home/mlokken/oriented_stacking/act_only_code/actmap_to_healpix.py {ymap_out} False none 4096 none")
'''
# make the number density maps if they do not exist
# this will make all of them
for i in range(len(omegam_varied)):
    OMstring = "OM{:d}{:s}".format(int(round(omegam_varied[i]*100)), addon)
    if constant_comoving:
        OMstring += '_const_comov'
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], H0=H0_varied[i], Ob0=omegab_varied[i])
    print("making number density maps")
    args = ["python", "cat_to_numden_maps_altcosmo.py", "{:.5f}".format(new_cosmo.Om0), "{:.5f}".format(new_cosmo.Ob0), "{:.5f}".format(new_cosmo.h), OMstring]
    print(args)
    subprocess.run(args)
stop

# get all peaks, stack
for i in range(len(omegam_varied)):
    OMstring = "OM{:d}{:s}".format(int(round(omegam_varied[i]*100)), addon)
    if constant_comoving:
        OMstring += '_const_comov'
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], H0=H0_varied[i], Ob0=omegab_varied[i])
    print("Getting peaks.")
    args = ["python", "getallpeaks_altcosmo.py", "{:.5f}".format(new_cosmo.Om0), "{:.5f}".format(new_cosmo.Ob0), "{:.5f}".format(new_cosmo.h), OMstring]
    subprocess.run(args)
    print("Stacking.")
    args = ["python", "stack_altcosmo.py", "{:.5f}".format(new_cosmo.Om0), "{:.5f}".format(new_cosmo.Ob0), "{:.5f}".format(new_cosmo.h), OMstring]
    subprocess.run(args)
    