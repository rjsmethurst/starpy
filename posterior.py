""" A fantastic python code to determine the quenched SFH parameters of galaxies using emcee (http://dan.iel.fm/emcee/current/). This file contains all the functions needed to determine the mean SFH parameters of a population.
    
    N.B. The data files .ised_ASCII contain the extracted bc03 models and have a 0 in the origin at [0,0]. The first row contains the model ages (from the second column) - data[0,1:]. The first column contains the model lambda values (from the second row) - data[1:,0]. The remaining data[1:,1:] are the flux values at each of the ages (columns, x) and lambda (rows, y) values 
    """

import numpy as N
import scipy as S
import pylab as P
import pyfits as F
import idlsave
import fluxes 
import pyfits as F
import emcee
import triangle
import time
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import kde
from scipy.interpolate import LinearNDInterpolator as i
from itertools import product

cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

font = {'family':'serif', 'size':25}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

"""We first define the directory in which we will find the BC03 model, extracted from the original files downloaded from the BC03 website into a usable format. Here we implement a solar metallicity model with a Chabrier IMF."""
dir ='/Users/becky/Projects/Green-Valley-Project/bc03/models/Padova1994/chabrier/ASCII/'
model = 'extracted_bc2003_lr_m62_chab_ssp.ised_ASCII'
data = N.loadtxt(dir+model)
n=0

print 'gridding...'
tq = N.linspace(0.003, 13.8, 100)
tau = N.linspace(0.003, 4, 100)
age = N.linspace(10.88861228, 13.67023409, 50)
grid = N.array(list(product(age, tau, tq)))
print 'loading...'
nuv_pred = N.load('nuv_look_up.npy')
ur_pred = N.load('ur_look_up.npy')
print 'functioning first...'
nuv_f = i (grid, nuv_pred)
print 'functioning second...'
ur_f = i(grid, ur_pred)

# Function which given a tau and a tq calculates the sfr at all times
def expsfh(tq, tau, time):
    """ This function when given a single combination of [tq, tau] values will calcualte the SFR at all times. First calculate the sSFR at all times as defined by Peng et al. (2010) - then the SFR at the specified tq of quenching and set the SFR as constant at this value before this time. Beyond this time the SFR is an exponentially declining function with timescale tau. 
        
        INPUT:
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. 
        
        :tq: 
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time.
        :time:
        An array of time values at which the SFR is calcualted at each step. 
        
        RETURNS:
        :sfr:
        Array of the same dimensions of time containing the sfr at each timestep.
        """
    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2)
    c_sfr = N.interp(tq, time, ssfr)*(1E10)/(1E9)
    # definition is for 10^10 M_solar galaxies and per gyr - convert to M_solar/year
    a = time.searchsorted(tq)
    sfr = N.ones(len(time))*c_sfr
    sfr[a:] = c_sfr*N.exp(-(time[a:]-tq)/tau)
    return sfr

def predict_c_one(theta, age):
    """ This function predicts the u-r and nuv-u colours of a galaxy with a SFH defined by [tq, tau], according to the BC03 model at a given "age" i.e. observation time. It calculates the colours at all times then interpolates for the observed age - it has to this in order to work out the cumulative mass across the SFH to determine how much each population of stars contributes to the flux at each time step. 
        
        :theta:
        An array of size (1,2) containing the values [tq, tau] in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5.
        
        :age:
        Observed age of a galaxy, often calculated from the redshift i.e. at z=0.1 the age ~ 12.5. Must be in units of Gyr. 
        
        RETURNS:
        :nuv_u_age:
        Array the same shape as :age: with the nuv-u colour values at each given age for the specified :theta: values
        
        :u_r_age:
        Array the same shape as :age: with the u-r colour values at each given age for the specified :theta: values
        """
    ti = N.arange(0, 0.01, 0.003)
    t = N.linspace(0,14.0,100)
    t = N.append(ti, t[1:])
    tq, tau = theta
    sfr = expsfh(tq, tau, t)
    # Work out total flux at each time given the sfh model of tau and tq (calls fluxes function)
    total_flux = fluxes.assign_total_flux(data[0,1:], data[1:,0], data[1:,1:], t*1E9, sfr)
    # Calculate fluxes from the flux at all times then interpolate to get one colour for the age you are observing the galaxy at - if many galaxies are being observed, this also works with an array of ages to give back an array of colours
    nuv_u, u_r = get_colours(t*1E9, total_flux, data)
    nuv_u_age = N.interp(age, t, nuv_u)
    u_r_age = N.interp(age, t, u_r)
    return nuv_u_age, u_r_age
    

def get_colours(time, flux, data):
    """" Calculates the colours of a given sfh fluxes across time given the BC03 models from the magnitudes of the SED.
        
        :time:
        Array of times at which the colours should be calculated. In units of Gyrs. 
        
        :flux:
        SED of fluxes describing the calcualted SFH. Returned from the assign_total_flux function in fluxes.py
        
        :data:
        BC03 model values for wavelengths, time steps and fluxes. The wavelengths are needed to calculate the magnitudes. 
        
        RETURNS:
        :nuv_u: :u_r:
        Arrays the same shape as :time: with the predicted nuv-u and u-r colours
        """
    nuvmag = fluxes.calculate_AB_mag(time, data[1:,0], flux, nuvwave, nuvtrans)
    umag = fluxes.calculate_AB_mag(time, data[1:,0], flux, uwave, utrans)
    rmag = fluxes.calculate_AB_mag(time, data[1:,0], flux, rwave, rtrans)
    nuv_u = nuvmag - umag
    u_r = umag - rmag
    return nuv_u, u_r

def lookup_col_one(theta, age):
    nuv_u = nuv_f(age, theta[1], theta[0])
    u_r = ur_f(age, theta[1], theta[0])
    return nuv_u, u_r


def lnlike_one(theta, ur, sigma_ur, nuvu, sigma_nuvu, age):
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
#    print 'theta: ', theta
#    print 'predicted nuvu: ', pred_nuvu
#    print 'obserbed nuvu: ', nuvu
#    print 'chi sq nuvu: ', -0.5*((nuvu-pred_nuvu)**2/sigma_nuvu**2)
#    print 'predicted ur: ', pred_ur
#    print 'observed ur: ', ur
#    print 'chi sq ur: ', -0.5*((ur-pred_ur)**2/sigma_ur**2)
    return -0.5*N.log(2*N.pi*sigma_ur**2)-0.5*((ur-pred_ur)**2/sigma_ur**2)-0.5*N.log10(2*N.pi*sigma_nuvu**2)-0.5*((nuvu-pred_nuvu)**2/sigma_nuvu**2)


def lnlike(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    """Function which takes the likelihood for ONE quenching model for both disc and smooth like galaxies and sums across all galaxies to return one value for a given set of theta = [tqd, taud, tqs, taus]. It also incorporates the morphological classifications from Galaxy Zoo for a smooth and disc galaxy. 
        
        :theta:
        An array of size (1,4) containing the values [tq, tau] for both smooth and dsic galaxies in Gyr.
        
        :tq:
        The time at which the onset of quenching begins in Gyr. Allowed ranges from the beginning to the end of known cosmic time. Can be either for smooth or disc galaxies.
        
        :tau:
        The exponential timescale decay rate of the star formation history in Gyr. Allowed range from the rest of the functions is 0 < tau [Gyr] < 5. Can be either for smooth or disc galaxies.
        
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
        
        :pd:
        Galaxy Zoo disc morphological classification debiased vote fraction
        
        :ps:
        Galaxy Zoo smooth morphological classification debiased vote fraction
        
        RETURNS:
        One value of the likelihood at the given :theta: summed over all the galaxies in the sample
        """
    ts, taus, td, taud = theta
    d = lnlike_one([td, taud], ur, sigma_ur, nuvu, sigma_nuvu, age)
    s = lnlike_one([ts, taus], ur, sigma_ur, nuvu, sigma_nuvu, age)
    D = N.log(pd) + d
    S = N.log(ps) + s
    return N.sum(N.logaddexp(D, S))

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
    ts, taus, td, taud = theta
    if 0.003 <= ts <= 13.807108309208775 and 0.003 <= taus <= 4.0 and 0.003 <= td < 13.807108309208775 and 0.003 <= taud <= 4.0:
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
    global n
    n+=1
    if n %100 == 0:
        print 'step number', n/100
    lp = lnprior(w, theta)
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps)

def sample(ndim, nwalkers, nsteps, burnin, start, w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
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
    if len(age) != len(ur):
        raise SystemExit('Number of ages does not coincide with number of galaxies...')
    p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=(w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps))
    # burn in run 
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/Users/becky/samples_burn_in_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
    N.save(samples_save, samples)
    walker_plot(samples, nwalkers, burnin)
    sampler.reset()
    print 'RESET', pos
    # main sampler run
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/Users/becky/samples_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
    N.save(samples_save, samples)
    fig = triangle.corner(samples, labels=[r'$ t_{smooth} $', r'$ \tau_{smooth} $', r'$ t_{disc} $', r'$ \tau_{disc}$'])
    fig.savefig('triangle_t_tau_gv_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')
    return samples, samples_save


#Define function to plot the walker positions as a function of the step
def walker_plot(samples, nwalkers, limit):
    """ Plotting function to visualise the steps of the walkers in each parameter dimension for smooth and disc theta values. 
        
        :samples:
        Array of shape (nsteps*nwalkers, 4) produced by the emcee EnsembleSampler in the sample function.
        
        :nwalkers:
        The number of walkers that step around the parameter space used to produce the samples by the sample function. Must be an even integer number larger than ndim.
        
        :limit:
        Integer value less than nsteps to plot the walker steps to. 
        
        RETURNS:
        :fig:
        The figure object
        """
    s = samples.reshape(nwalkers, -1, 4)
    s = s[:,:limit, :]
    fig = P.figure(figsize=(8,10))
    ax1 = P.subplot(4,1,1)
    ax2 = P.subplot(4,1,2)
    ax3 = P.subplot(4,1,3)
    ax4 = P.subplot(4,1,4)
    for n in range(len(s)):
        ax1.plot(s[n,:,0], 'k')
        ax2.plot(s[n,:,1], 'k')
        ax3.plot(s[n,:,2], 'k')
        ax4.plot(s[n,:,3], 'k')
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='x', labelbottom='off')
    ax4.set_xlabel(r'step number')
    ax1.set_ylabel(r'$t_{smooth}$')
    ax2.set_ylabel(r'$\tau_{smooth}$')
    ax3.set_ylabel(r'$t_{disc}$')
    ax4.set_ylabel(r'$\tau_{disc}$')
    P.subplots_adjust(hspace=0.1)
    save_fig = '/Users/becky/walkers_steps_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
    fig.savefig(save_fig)
    return fig

def corner_plot(s, labels, extents, bf):
    """ Plotting function to visualise the gaussian peaks found by the sampler function. 2D contour plots of tq against tau are plotted along with kernelly smooth histograms for each parameter.
        
        :s:
         Array of shape (#, 2) for either the smooth or disc produced by the emcee EnsembleSampler in the sample function of length determined by the number of walkers which resulted at the specified peak.
        
        :labels:
        List of x and y axes labels i.e. disc or smooth parameters
        
        :extents:
        Range over which to plot the samples, list shape [[xmin, xmax], [ymin, ymax]]
        
        :bf:
        Best fit values for the distribution peaks in both tq and tau found from mapping the samples. List shape [(tq, poserrtq, negerrtq), (tau, poserrtau, negerrtau)]
        
        RETURNS:
        :fig:
        The figure object
        """
    x, y = s[:,0], s[:,1]
    fig = P.figure(figsize=(6.25,6.25))
    ax2 = P.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    triangle.hist2d(x, y, ax=ax2, bins=100, extent=extents, plot_contours=True)
    ax2.axvline(x=bf[0][0], linewidth=1)
    ax2.axhline(y=bf[1][0], linewidth=1)
    [l.set_rotation(45) for l in ax2.get_xticklabels()]
    [j.set_rotation(45) for j in ax2.get_yticklabels()]
    ax2.tick_params(axis='x', labeltop='off')
    ax1 = P.subplot2grid((3,3), (0,0),colspan=2)
    #ax1.hist(x, bins=100, range=(extents[0][0], extents[0][1]), normed=True, histtype='step', color='k')
    den = kde.gaussian_kde(x[N.logical_and(x>=extents[0][0], x<=extents[0][1])])
    pos = N.linspace(extents[0][0], extents[0][1], 750)
    ax1.plot(pos, den(pos), 'k-', linewidth=1)
    ax1.axvline(x=bf[0][0], linewidth=1)
    ax1.axvline(x=bf[0][0]+bf[0][1], c='b', linestyle='--')
    ax1.axvline(x=bf[0][0]-bf[0][2], c='b', linestyle='--')
    ax1.set_xlim(extents[0][0], extents[0][1])
    #    ax12 = ax1.twiny()
    #    ax12.set_xlim((extent[0][0], extent[0][1])
    #ax12.set_xticks(N.array([1.87, 3.40, 6.03, 8.77, 10.9, 12.5]))
    #ax12.set_xticklabels(N.array([3.5, 2.0 , 1.0, 0.5, 0.25, 0.1]))
    #    [l.set_rotation(45) for l in ax12.get_xticklabels()]
    #    ax12.tick_params(axis='x', labelbottom='off')
    #    ax12.set_xlabel(r'$z$')
    ax1.tick_params(axis='x', labelbottom='off', labeltop='off')
    ax1.tick_params(axis='y', labelleft='off')
    ax3 = P.subplot2grid((3,3), (1,2), rowspan=2)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    #ax3.hist(y, bins=100, range=(extents[1][0], extents[1][1]), normed=True, histtype='step',color='k', orientation ='horizontal')
    den = kde.gaussian_kde(y[N.logical_and(y>=extents[1][0], y<=extents[1][1])])
    pos = N.linspace(extents[1][0], extents[1][1], 750)
    ax3.plot(den(pos), pos, 'k-', linewidth=1)
    ax3.axhline(y=bf[1][0], linewidth=1)
    ax3.axhline(y=bf[1][0]+bf[1][1], c='b', linestyle='--')
    ax3.axhline(y=bf[1][0]-bf[1][2], c='b', linestyle='--')
    ax3.set_ylim(extents[1][0], extents[1][1])
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    P.tight_layout()
    return fig


""" Load the magnitude bandpass filters using idl save """
filters = idlsave.read('/Users/becky/Projects/Green-Valley-Project/kevin_idl/ugriz.sav')
fuvwave= filters.ugriz.fuvwave[0]
fuvtrans = filters.ugriz.fuvtrans[0]
nuvwave= filters.ugriz.nuvwave[0]
nuvtrans = filters.ugriz.nuvtrans[0]
uwave= filters.ugriz.uwave[0]
utrans = filters.ugriz.utrans[0]
gwave= filters.ugriz.gwave[0]
gtrans = filters.ugriz.gtrans[0]
rwave= filters.ugriz.rwave[0]
rtrans = filters.ugriz.rtrans[0]
iwave= filters.ugriz.iwave[0]
itrans = filters.ugriz.itrans[0]
zwave= filters.ugriz.zwave[0]
ztrans = filters.ugriz.ztrans[0]
vwave= filters.ugriz.vwave[0]
vtrans = filters.ugriz.vtrans[0]
jwave= filters.ugriz.jwave[0]
jtrans = filters.ugriz.jtrans[0]
hwave= filters.ugriz.hwave[0]
htrans = filters.ugriz.htrans[0]
kwave= filters.ugriz.kwave[0]
ktrans = filters.ugriz.ktrans[0]
