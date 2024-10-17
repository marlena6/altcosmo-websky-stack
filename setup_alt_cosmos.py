####################################################
# Notes on the Niagara setup for runs
# Probably due to high memory issues, ntasks=27 is not working
# ntasks=8 worked, on one node.
# probably the ntasks has to be smaller for bigger runs.
# so that it doesn't max out the memory/node on Niagara.
# dividing into multiple nodes e.g. ntasks=9, nnodes = 3 does not work
#####################################################


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
import camb
from camb import model, initialpower
from colossus.cosmology import cosmology


# set standard cosmology (Planck 18)
P18 = cosmology.setCosmology('planck18-only') # consistent with the Peak Patch setup
P18_omegam = P18.Om0
P18_omegab = P18.Ob0
omegak = P18.Ok0
print("Fiducial omega_M:", P18_omegam)
# set values of Omega_M
omegam_varied = np.delete(np.linspace(P18_omegam-0.1,P18_omegam+0.1,5), 2) # delete the standard case
print("Omega_M will be:", omegam_varied)

# get corresponding values of Omega_Lambda
omegal_varied = 1-omegam_varied
print("Omega_L will be:", omegal_varied)
print("matter-DE equality will be at:", 1/(omegam_varied/omegal_varied)**(1/3)-1)

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

basedir = "/home/r/rbond/lokken/scratch/peak-patch-runs/"

addon = ''

if fix_density:
    addon += '_fix_physdens'
    omh2 = P18_omegam*P18_h**2
    obh2 = P18_omegab*P18_h**2
    omegab_varied = omegam_varied*P18_omegab/P18_omegam # fix the baryon fraction
    print("Planck18 Omega m h^2: ", omh2)
    print("Planck18 Omega b h^2: ", obh2)
    print("Omegab", omegab_varied)
    h_varied = np.sqrt(omh2/omegam_varied)
    omegac_varied = omegam_varied-omegab_varied
    print("Omega_cdm", omegac_varied)
    H0_varied = 100*h_varied
    print("H0 varied: ", h_varied)
    print("Omega_b/Omega_m is:", omegab_varied/omegam_varied)

    
else:
    # get Omega_b to maintain the baryon fraction
    omegac_varied = omegam_varied-P18_omegab
    print("Omega_b/Omega_m is:", P18_omegab/omegam_varied)
    h_varied = np.full(len(omegam_varied), P18_h) # fix the hubble constant
    H0_varied = 100*h_varied
    omegab_varied = np.full(len(omegam_varied), P18_omegab) # fix the baryon fraction


for i in range(len(omegam_varied)):
# for i in [2]: # only do it for standard case
    print("Omega M = {:.2f}".format(omegam_varied[i]))
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], H0=H0_varied[i], Ob0=omegab_varied[i])
    D = new_cosmo.growthFactor(z[0])
    sigma8_nonfid_z0 = sigma8/D # how the fiducial sigma 8 would grow to today, this accounts for the rescaling
    psfile_chosen = '/home/r/rbond/lokken/software/peakpatch/tables/power_OmegaM_{:.2f}{:s}.dat'.format(omegam_varied[i],addon)
    psfile_chosen = psfile_chosen.split('/')[-1] # get the filename only
    print(psfile_chosen)
    dirname = basedir+'OmegaM_{:.2f}/'.format(omegam_varied[i])
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)
    if not os.path.isdir(dirname+'param'):
        os.mkdir('param')
    if os.path.exists(dirname+'param/param.params'):
        os.remove('param/param.params')
    fout = open(dirname+'param/param.params', 'w')
    with open(basedir+"standard_param.params") as f:
        for line in f:
            if line.startswith("Omx"):
                fout.write(line.replace(line, "Omx          = {:.5f}  # Omega_c, the density fraction of CDM\n".format(new_cosmo.Om0-new_cosmo.Ob0)))
            elif line.startswith("sigma8"):
                fout.write(line.replace(line, "sigma8       = {:.5f}  # sigma(z=0,r=8 Mpc)\n".format(sigma8_nonfid_z0)))
            elif line.startswith("pkfile"):
                fout.write(line.replace(line, "pkfile       = '{:s}'\n".format(psfile_chosen)))
            elif line.startswith("h      "):
                fout.write(line.replace(line, "h            = {:.5f}\n".format(new_cosmo.h)))
            elif line.startswith("OmB"):
                fout.write(line.replace(line, "OmB          = {:.5f}  # Omega_b, the density fraction of baryons\n".format(new_cosmo.Ob0)))
            elif line.startswith("run_name"):
                fout.write(line[:-2]+'_OM{:d}{:s}\'\n'.format(int(round(omegam_varied[i]*100)), addon))
            elif line.startswith("short_name"):
                fout.write(line[:-2]+'_OM{:d}{:s}\'\n'.format(int(round(omegam_varied[i]*100)), addon))
            else:
                fout.write(line)
    fout.close()
    f.close()
    
    print('running system commmands.')
    os.system('peak-patch.py param/param.params')
    os.system('./bin/hpkvd 1')
    os.system('sbatch 2400Mpc_nb28_OM{:d}{:s}_13579.sh'.format(int(round(omegam_varied[i]*100)), addon))
    os.chdir(basedir)
