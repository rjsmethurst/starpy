from posterior import *
from astropy.cosmology import FlatLambdaCDM

import numpy as N
import pylab as PY
import pyfits as F
import os
import time

""" The main control hub of the StarfPy package - load your data here and implement the funcions in the 'posterior.py' file. Originally the data is loaded with PyFITS - if you download the Galaxy Zoo data from the website it will be in this form. This code is set up to run on each of the galaxies in your data file one at a time in a for loop - each galaxy takes about 3 minutes to run on a laptop using the lookup table functions - this will take ~ months for 100K+ galaxies. You have been warned. Use a supercomputer for large numbers of galaxies.
    """

font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

### Use PyFits to open up the Galaxy Zoo data
file = str(raw_input('Location of the FITS file containing your galaxy data : '))
dat = F.open(file)
gz2data = dat[1].data
dat.close()

col = N.zeros(16*len(gz2data)).reshape(len(gz2data), 16)
col[:,0] = gz2data.field('MU_MR')
col[:,1] = gz2data.field('NUV_U')
col[:,2] = gz2data.field('t01_smooth_or_features_a01_smooth_debiased')
col[:,3] = gz2data.field('t01_smooth_or_features_a02_features_or_disk_debiased')
col[:,4] = ((gz2data.field('Err_MU_MR'))**2 + 0.05**2)**0.5
col[:,5] = ((gz2data.field('Err_NUV_U'))**2 + 0.1**2)**0.5
col[:,6] = gz2data.field('z_1')
col[:,7] = gz2data.field('zErr_1')
col[:,8] = gz2data.field('GV_first')
col[:,9] = gz2data.field('GV_sec')
col[:,10] = gz2data.field('upper_GV')
col[:,11] = gz2data.field('lower_GV')
col[:,12] = gz2data.field('dr7objid')
col[:,13] = gz2data.field('dr8objid')
col[:,14] = gz2data.field('ra_1')
col[:,15] = gz2data.field('dec_1')

# Remove NaN values from data
non_nan = N.logical_not(N.isnan(col[:,1])).astype(int)
data = N.compress(non_nan, col, axis=0)

age_save = str(raw_input('Desired or current location of galaxy ages from astropy.cosmology, e.g. "~/starfpy/galaxy_data_ages.npy" : '))
age_path = os.path.exists(age_save)
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)
if age_path ==False:
    age = N.array(cosmo.age(data[:,6]))
    N.save(age_save, age)
else:
    age = N.load(age_save)
print len(age)

nwalkers = 100
nsteps= 400
start = [7.5, 1.5]
burnin = 400

start_time = time.time()
#The rest calls the emcee module through the 'posterior.py' functions and makes a plot....
for n in range(len(data)):
    url = 'http://casjobs.sdss.org/ImgCutoutDR7/getjpeg.aspx?ra='+str(data[n,14])+'&dec='+str(data[n,15])+'&scale=0.099183&width=424&height=424'
    f = wget.download(url, out=str(int(data[n,13]))+'.jpeg')
    samples, ss = sample(2, nwalkers, nsteps, burnin, start, data[n,0], data[n,4], data[n, 1], data[n, 5], age[n], data[n,3], data[n,2], data[n,13])
    elap = (time.time() - start_time)/60
    print 'Minutes taken for '+str(len(samples)/nwalkers
                               )+' steps and '+str(nwalkers)+' walkers for the '+str(n)+'th galaxy in the data', elap
    tq_mcmc, tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [16,50,84],axis=0)))
    print 'tq_smooth',tq_mcmc
    print 'tau_smooth',tau_mcmc

    fig = corner_plot(samples, labels = [r'$ t_{quench}$', r'$ \tau$'], extents=[[N.min(samples[:,0]), N.max(samples[:,0])],[N.min(samples[:,1]),N.max(samples[:,1])]], bf=[tq_mcmc, tau_mcmc], id=data[n,13])


