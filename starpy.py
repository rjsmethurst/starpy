from posterior import *
from astropy.cosmology import FlatLambdaCDM
import numpy as N
import sys

# Use sys to assign arguments for the galaxy data from the command line
u_r, err_u_r, nuv_u, err_nuv_u, z, pd, ps, dr8, ra, dec = sys.argv[1:]

# Use astropy to calculate the age from the redshift in the data 
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)
age = N.array(cosmo.age(float(z)))

# Define parameters needed for emcee 
nwalkers = 100 # number of monte carlo chains
nsteps= 400 # number of steps in the monte carlo chain
start = [7.5, 1.5] # starting place of all the chains
burnin = 400 # number of steps in the burn in phase of the monte carlo chain

#The rest calls the emcee module which is initialised in the sample function of the posterior file. 
s = sample(2, nwalkers, nsteps, burnin, start, float(u_r), float(err_u_r), float(nuv_u), float(err_nuv_u), age, float(pd), float(ps), dr8, ra, dec)
tq_mcmc, tau_mcmc,  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))
fig = corner_plot(samples, labels = [r'$ t_{quench}$', r'$ \tau$'], extents=[[N.min(s[:,0]), N.max(s[:,0])],[N.min(s[:,1]),N.max(s[:,1])]], bf=[tq_mcmc, tau_mcmc], id=dr8)
fig.savefig('starpy_output_'+str(dr8)+'.pdf')
print 'Best fit [t, tau] values found by starpy for input parameters are : [', tq_mcmc[0], tau_mcmc[0], ']'
