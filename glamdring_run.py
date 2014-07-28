""" A fantastic python code to determine the quenched SFH parameters of galaxies using emcee (http://dan.iel.fm/emcee/current/). This file contains all the functions needed to determine the mean SFH parameters of a population.
    
    N.B. The data files .ised_ASCII contain the extracted bc03 models and have a 0 in the origin at [0,0]. The first row contains the model ages (from the second column) - data[0,1:]. The first column contains the model lambda values (from the second row) - data[1:,0]. The remaining data[1:,1:] are the flux values at each of the ages (columns, x) and lambda (rows, y) values 
    """

import numpy as N
import emcee
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp2d
from itertools import product

# first define parameters that will be used by all of the runs - including loading the model look up tables to which the emcee function will try to match the data 
ages = N.linspace(10.88861228, 13.67023409, 50)
tau = N.linspace(0.003, 4, 100)
tq = N.linspace(0.003, 13.8, 100)
grid = N.array(list(product(ages, tau, tq)))
nuv = N.load('nuv_look_up_ssfr.npy', 'r')
ur = N.load('ur_look_up_ssfr.npy', 'r')
lu = N.append(nuv.reshape(-1,1), ur.reshape(-1,1), axis=1)
# remove unnecessary tables from memory
del nuv
del ur

def lookup_col_one(theta, age):
    """ Function for determining the predicted colours for give model parameters, theta, using the look up tables loaded previously. 
        :theta: 
        An array of size (1,2) containing the values [tq, tau] in Gyr.
        
        """
    ur_pred = u(theta[0], theta[1])
    nuv_pred = v(theta[0], theta[1])
    return nuv_pred, ur_pred


def lnlike_one(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    """ Function for determining the likelihood of ONE quenching model described by theta = [tq, tau] for all the galaxies in the sample. Simple chi squared likelihood between predicted and observed colours of the galaxies. 
        
        :theta:
        An array of size (1,2) containing the values [tq, tau] in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5.
        
        :ur:
        Observed u-r colour of a galaxy; k-corrected.
        
        :sigma_ur:
        Error on the observed u-r colour of a galaxy
        
        :nuvu:
        Observed nuv-u colour of a galaxy; k-corrected.
        
        :sigma_nuvu:
        Error on the observed nuv-u colour of a galaxy
        
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr.
        
        RETURNS:
        Array of same shape as :age: containing the likelihood for each galaxy at the given :theta:
        """
    tq, tau = theta
    pred_nuvu, pred_ur = lookup_col_one(theta, age)
    return -0.5*N.log(2*N.pi*sigma_ur**2)-0.5*((ur-pred_ur)**2/sigma_ur**2)-0.5*N.log10(2*N.pi*sigma_nuvu**2)-0.5*((nuvu-pred_nuvu)**2/sigma_nuvu**2)


# Prior likelihood on theta values given the inital w values assumed for the mean and stdev
def lnprior(w, theta):
    """ Function to calcualted the prior likelihood on theta values given the inital w values assumed for the mean and standard deviation of the tq and tau parameters. Defined ranges are specified - outside these ranges the function returns -N.inf and does not calculate the posterior probability. 
        
        :w:
        Prior assumptions on the distribution of theta for disc and smooth galaxies. Assumed normal distribution for all parameters.
        
        :theta: 
        An array of size (1,4) containing the values [tq, tau] for both smooth and disc galaxies in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Can be either for smooth or disc galaxies.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Can be either for smooth or disc galaxies.
        
        RETURNS:
        Value of the prior at the specified :theta: value.
        """
    mu_tqs, mu_taus, mu_tqd, mu_taud, sig_tqs, sig_taus, sig_tqd, sig_taud = w
    tq, tau = theta
    if 0.003 <= tq <= 13.807108309208775 and 0.003 <= tau <= 4.0:
        return 0.0
    else:
        return -N.inf

# Overall likelihood function combining prior and model
def lnprob(theta, w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    """Overall posterior function combiningin the prior and calculating the likelihood. Also prints out the progress through the code with the use of n. 
        
        :theta:
        An array of size (1,4) containing the values [tq, tau] for both smooth and disc galaxies in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Can be either for smooth or disc galaxies.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Can be either for smooth or disc galaxies.
        
        :w:
        Prior assumptions on the distribution of theta for disc and smooth galaxies. Assumed normal distribution for all parameters.
        
        :ur:
        Observed u-r colour of a galaxy; k-corrected. An array of shape (N,1) or (N,).
        
        :sigma_ur:
        Error on the observed u-r colour of a galaxy. An array of shape (N,1) or (N,).
        
        :nuvu:
        Observed nuv-u colour of a galaxy; k-corrected. An array of shape (N,1) or (N,).
        
        :sigma_nuvu:
        Error on the observed nuv-u colour of a galaxy. An array of shape (N,1) or (N,).
        
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (N,1) or (N,).
        
        :pd:
        Galaxy Zoo disc morphological classification debiased vote fraction. An array of shape (N,1) or (N,).
        
        :ps:
        Galaxy Zoo smooth morphological classification debiased vote fraction. An array of shape (N,1) or (N,).
        
        RETURNS:
        Value of the posterior function for the given :theta: value.
        
        """
    lp = lnprior(w, theta)
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike_one(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps)

def sample(ndim, nwalkers, nsteps, burnin, start, w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps, dr8):
    """ Function to implement the emcee EnsembleSampler function for the sample of galaxies input. Burn in is run and calcualted fir the length specified before the sampler is reset and then run for the length of steps specified. 
        
        :ndim:
        The number of parameters in the model that emcee must find. In this case it always 4 with tqs, taus, tqd, taud.
        
        :nwalkers:
        The number of walkers that step around the parameter space. Must be an even integer number larger than ndim. 
        
        :nsteps:
        The number of steps to take in the final run of the MCMC sampler. Integer.
        
        :burnin:
        The number of steps to take in the inital burn-in run of the MCMC sampler. Integer. 
        
        :start:
        The positions in the tq and tau parameter space to start for both disc and smooth parameters. An array of shape (1,4).
        
        :w:
        Prior assumptions on the distribution of theta for disc and smooth galaxies. Assumed normal distribution for all parameters.
        
        :ur:
        Observed u-r colour of a galaxy; k-corrected. An array of shape (N,1) or (N,).
        
        :sigma_ur:
        Error on the observed u-r colour of a galaxy. An array of shape (N,1) or (N,).
        
        :nuvu:
        Observed nuv-u colour of a galaxy; k-corrected. An array of shape (N,1) or (N,).
        
        :sigma_nuvu:
        Error on the observed nuv-u colour of a galaxy. An array of shape (N,1) or (N,).
        
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. An array of shape (N,1) or (N,).
        
        :pd:
        Galaxy Zoo disc morphological classification debiased vote fraction. An array of shape (N,1) or (N,).
        
        :ps:
        Galaxy Zoo smooth morphological classification debiased vote fraction. An array of shape (N,1) or (N,).
        
        RETURNS:
        :samples:
        Array of shape (nsteps*nwalkers, 4) containing the positions of the walkers at all steps for all 4 parameters.
        :samples_save:
        Location at which the :samples: array was saved to. 
        
        """
    # start by creating a 2D look-up table for a single galaxy age by interpolating over a splice of the 3D look-up table lu
    global u
    global v
    a = N.searchsorted(ages, age)
    b = N.array([a-1, a])
    # splice 3d array to give two 2d arrays at the ages bounding the single galaxy age 
    g = grid[N.where(N.logical_or(grid[:,0]==ages[b[0]], grid[:,0]==ages[b[1]]))]
    val = lu[N.where(N.logical_or(grid[:,0]==ages[b[0]], grid[:,0]==ages[b[1]]))]
    # interpolate over these two 2d arrays 
    f = LinearNDInterpolator(g, val, fill_value=(-N.inf))
    # generate single 2d look up table for single galaxy age 
    look = f(age, g[:10000, 1], g[:10000, 2])
    # create function for interpolating over for first colour to match to
    v = interp2d(tq, tau, look[:,0].reshape(100,100))
    # create function for interpolating over for second colour to match to
    u = interp2d(tq, tau, look[:,1].reshape(100,100))
    # set up start positions for emcee walkers
    p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps))
    # complete the burn in run
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    # SAVE the positions of the burn in chain
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/usersVol1/smethurst/hyper/burnin/samples_burn_in_'+str(int(dr8))+'.npy'
    N.save(samples_save, samples)
    # reset the mcmc chain and start the run from the last position of the burn in
    sampler.reset()
    # complete main sampler run
    sampler.run_mcmc(pos, nsteps)
    # SAVEV the positions of the mcmc chain 
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/usersVol1/smethurst/hyper/samples/samples_'+str(int(dr8))+'.npy'
    N.save(samples_save, samples)
    # remove variables from memory for good measure
    del u
    del v
    del f
    del g
    del look
    del a
    del b
    del samples
    del samples_save
