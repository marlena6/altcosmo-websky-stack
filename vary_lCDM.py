# vary_lCDM.py

###########################################################################
##  This script uses CAMB (Code for Anisotropies in the Microwave        ##
##  Background) to make tables of power spectra and transfer functions   ##
##  for a range of LambdaCDM cosmologies to construct initial condition  ##
##  fields for Peak Patch.                                               ##
##  Modified from Peak Patch code by George Stein and Nate Carlson       ##
###########################################################################

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
omegam_varied = np.linspace(P18_omegam-0.1,P18_omegam+0.1,5)
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


#######################################################s####################
##  Initialize CAMB calculation of power spectra and transfer functions  ##
##  by creating an object of class CAMBparams, pars, and setting it to   ##
##  the values corresponding to our cosmology.                           ##
###########################################################################


check_fiducial = False # test that the growth function is producing correct z=0 cosmology

def get_camb_ps(colossus_cosmo, z):
    # Set cosmological parameters for CAMB power spectra, fid at z=0, for comparison
    pars = camb.CAMBparams() # Make parameter object
    h    = colossus_cosmo.H0/100.
    # Range of wavenumbers to be used
    kmax_h   = 5.0e3/h  # max wavenumber / h
    kmin_h   = 5.0e-6/h # min wavenumber / h
    nkpoints = 1000     # number of points in power spectra tables
    pars.set_cosmology(H0=colossus_cosmo.H0, ombh2=colossus_cosmo.Ob0*h**2, omch2=(colossus_cosmo.Om0-colossus_cosmo.Ob0)*h**2,
        mnu=m_nu, omk=colossus_cosmo.Ok0, tau=tau) # Set with above cosmology
    pars.set_dark_energy() # re-set default DE equation of state
    pars.InitPower.set_params(ns=ns, As=As) # sets primordial power spectrum

    # Set parameters for calculating linear matter power spectrum at redshift z
    # and maximum wavenumber kmax_h*h
    pars.set_matter_power(redshifts=[z], kmax=kmax_h*h)
    pars.NonLinear = model.NonLinear_none # linear matter power spectrum only
    pars.PK_WantTransfer = 1 # Tell CAMB to calculate the matter
    pars.WantTransfer    = 1 #     power transfer function
    pars.Transfer.kmax   = kmax_h*h # Maximum wavenumber for transfer function

    results = camb.get_results(pars) # object of class CAMBdata

    # Calcualte matter power spectrum for CAMBdata object results
    kh, zz, pk = results.get_matter_power_spectrum(minkh=kmin_h,
        maxkh=kmax_h, npoints=nkpoints)
    return(results, kh, zz, pk)


for i in range(len(omegam_varied)):
    if check_fiducial:
        i+=2 # switch to the fiducial cosmology, code will break later
    
    # make the colossus cosmology object
    new_cosmo = cosmology.setCosmology('newcosmo', params=cosmology.cosmologies['planck18-only'], Om0=omegam_varied[i], H0=H0_varied[i], Ob0=omegab_varied[i])
    # make sure dark energy is changing, all others are fixed
    print("New cosmo parameters: Lambda density, H0, OmegaM, Omegab, ", new_cosmo.Ode0,  new_cosmo.H0, new_cosmo.Om0, new_cosmo.Ob0)
    h = new_cosmo.H0/100.
    print("Omega m h^2, Omega b h^2, Omega lambda h^2: ", new_cosmo.Om0*h**2, new_cosmo.Ob0*h**2, new_cosmo.Ode0*h**2)
    # Get sigma_8 (standard deviation for overdensity smoothed at 8 Mpc/h) value.
    sigma8_nonfid = new_cosmo.sigma(R=8, z=0.5)
    print("Varied cosmo sigma8 at z=0.5", sigma8_nonfid)
    '''
    # Set cosmological parameters for CAMB power spectra in above cosmology
        
    results_z5_var, kh, zz_z5_var, pk_z5_var = get_camb_ps(new_cosmo, 0.5)
    
    ###########################################################################
    ##  Compute Transfer function T()                                   ##
    ###########################################################################

    # Calculate matter transfer function
    transfer_z5_var = results_z5_var.get_matter_transfer_data()

    # Transfer functions as NumPy array Tf[n,i,j]
    Tf_z5_var = transfer_z5_var.transfer_data # transfer funcs as function of z and q-modes
    # divided by k^2 so that they are roughly constant at low k on super-
    # horizon scales
    # where n=6 is the total matter power, index i corresponds to the index of
    # q modes calculated, and j corresponds to the index of redshift z[j]
    kk = transfer_z5_var.q/h # q mode over h
    Tf_m_z5_var = Tf_z5_var[6,:,0]  # total matter power at z=0

    # Normalize matter power spectrum at z=0.5
    

    norm = (sigma8/sigma8_nonfid)**2 # normalization constant
    # this will even scale the 'fiducial' cosmology a little because there's something up 
    # with the CAMB calculation, sigma8 is too high
    print("norm", norm)
    k    = kh * h         # exact wavenumber k, not scaled by h
    pk_pre_rescale_z5_var = pk_z5_var[0,:] / ( 2.*np.pi * h)**3 # normalized P_m(z=0.5,k)
    pk_z5_var_rescale  = norm * pk_pre_rescale_z5_var # normalized P_m(z=0.5,k)
    # now find the power spectrum at z=0 by the growth factor
    
    D = new_cosmo.growthFactor(0.5)
    print("growth factor", D)
    pk_z0_var_rescale = pk_z5_var_rescale/D**2
    
    # Note here that we divide by h^3 to get pk in units of (Mpc/h)^3
    colors = ['red', 'green', 'black', 'purple', 'grey']
    plt.loglog(kh,pk_z5_var_rescale*( 2.*np.pi * h)**3, label='Om={:.2f}, p(k,z=0.5)'.format(omegam_varied[i]), color=colors[0], linestyle = 'solid')
    plt.loglog(kh,pk_z0_var_rescale*( 2.*np.pi * h)**3, label='p(k,z=0.0)', color=colors[1], linestyle='dashed')
    if check_fiducial:
        # get the fiducial P(k)(0) from CAMB
        # Calcualte matter power spectrum for CAMBdata object results
        results_fid_z0, kh_fid_z0, zz_fid_z0, pk_fid_z0 = get_camb_ps(P18, 0)
        pk_z0_camb = pk_fid_z0[0,:] / (2.*np.pi * P18_h)**3
        plt.loglog(kh,pk_z0_camb*( 2.*np.pi * P18_h)**3, label='p(k,z=0.0), fiducial', color='magenta', linestyle='dotted')
    plt.ylim([1, 10**5])
    plt.xlim([10**-4, 10])
    
    if check_fiducial:
        plt.legend()
        plt.savefig("ps_omegam_fid_test.png")
        break
    
    # make sure that the new power spectrum at z=0 is different than usually
    
    results_z0_var, kh_z0_var, zz_z0_var, pk_z0_var = get_camb_ps(new_cosmo,0)
    plt.loglog(kh_z0_var, pk_z0_var[0,:], label='Om={:.2f}, p(k,z=0.0 w/o rescale)'.format(omegam_varied[i]), color=colors[3], linestyle = 'dotted')
    plt.loglog(kh_z0_var,pk_pre_rescale_z5_var*( 2.*np.pi * h)**3, label='Om={:.2f}, p(k,z=0.5) w/o rescale'.format(omegam_varied[i]), color=colors[4], linestyle = 'solid')
    
    plt.legend()
    plt.savefig('ps_omegam_{:.2f}{:s}.png'.format(omegam_varied[i], addon))
    plt.clf()
    
    # Calculate mass and sigma
    
    # Primordial zeta power spectrum
    ko     = 0.05
    pkzeta = 2*np.pi**2*As/k**3 * (k/ko)**(ns-1)

    # Light field power spectra
    Achi = (5.e-7)**2
    pkchi = 2*np.pi**2*Achi/k**3 #in units of sigmas
    pkchi = pkchi/(2*np.pi)**3 #for pp power spectra
    #Get transfer function
    Trans = np.sqrt(pk_z0_var_rescale/pkzeta)

    # New model (Nate, 22 April 2022) spatially localized intermittent non
    # -Gaussianity
    #A2     = 1.6e-19 
    #R2     = 6.4e-1  # Mpc/h
    #pkchi2 = A2*(k*R2)**2 * np.exp( -k**2*R2**2 )
    #pkchi2 = 2*np.pi**2*k**-3 * pkchi2

    #R   = (10**Ma*3/4/np.pi/3.4e10)**(1./3)
    
    
    np.savetxt("/home/r/rbond/lokken/software/peakpatch/tables/power_OmegaM_%.2f%s.dat"%(omegam_varied[i], addon) ,np.transpose([k,pk_z0_var_rescale,Trans,pkchi]),fmt='%1.4e')
    #np.savetxt("power.dat",np.transpose([k,pk,Trans,pkchi,pkchi2]),fmt='%1.4e')
    #np.savetxt("sigma.dat",np.transpose([Ma,sigma]),fmt='%1.4e')



'''