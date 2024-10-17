import healpy as hp
import numpy as np
# load in websky map

nside = 4096
allpix = np.arange(hp.nside2npix(nside))
ra, dec = hp.pix2ang(nside, allpix, lonlat=True)

# load in websky map
# eighth of sky
lonra_mask = [0,90]
latra_mask = [0,70]


allpix = np.arange(hp.nside2npix(nside))
ra,dec = hp.pix2ang(nside, allpix, lonlat=True)
maprange = (ra>lonra_mask[0]) & (ra<lonra_mask[1]) & (dec>latra_mask[0]) & (dec<latra_mask[1])
inrange = allpix[maprange]
testmap = np.zeros(len(allpix))
testmap[inrange]=1
hp.write_map("/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup.fits", testmap, overwrite=True)

# make a mask that is 5 degrees reduced from the edges, for clusters
reduce = 5
lonra_cl1 = lonra_mask[1]-reduce
lonra_cl0 = lonra_mask[0]+reduce
latra_cl1 = latra_mask[1]-reduce
latra_cl0 = latra_mask[0]+reduce
allpix = np.arange(hp.nside2npix(nside))
ra,dec = hp.pix2ang(nside, allpix, lonlat=True)
maprange = (ra>lonra_cl0) & (ra<lonra_cl1) & (dec>latra_cl0) & (dec<latra_cl1)
inrange = allpix[maprange]
testmap = np.zeros(len(allpix))
testmap[inrange]=1
hp.write_map("/mnt/raid-cita/mlokken/pkpatch/eighthsky_mask_nopole_midleftup_reduced{:d}.fits".format(reduce), testmap, overwrite=True)