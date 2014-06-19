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
from scipy import linalg
from itertools import product
from scipy.interpolate import griddata
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import kde

cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

font = {'family':'serif', 'size':20}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

""" The data files .ised_ASCII contain the extracted bc03 models and have a 0 in the origin at [0,0]. The first row contains the model ages (from the second column) - data[0,1:]. The first column contains the model lambda values (from the second row) - data[1:,0]. The remaining data[1:,1:] are the flux values at each of the ages (columns, x) and lambda (rows, y) values 
"""



dir ='/Volumes/Data/smethurst/Green-Valley-Project/bc03/models/Padova1994/chabrier/ASCII/'
model = 'extracted_bc2003_lr_m62_chab_ssp.ised_ASCII'
data = N.loadtxt(dir+model)
n=0

# Function which given a tau and a tq calculates the sfr at all times
def expsfh(tau, tq, time):
    ssfr = 2.5*(((10**10.27)/1E10)**(-0.1))*(time/3.5)**(-2.2) #ssfr as defined by Peng et al (2010)
    c_sfr = N.interp(tq, time, ssfr)*(1E10)/(1E9) # definition is for 10^10 M_solar galaxies and per gyr - convert to M_solar/year
    a = time.searchsorted(tq)
    sfr = N.ones(len(time))*c_sfr
    sfr[a:] = c_sfr*N.exp(-(time[a:]-tq)/tau)
    return sfr

# predict the colour of a galaxy of a given age given a sf model of tau and tq
def predict_c_one(theta, age):
    # Time, tq and tau are in units of Gyrs
    ti = N.arange(0, 0.01, 0.003)
    t = N.linspace(0,14.0,100)
    t = N.append(ti, t[1:])
    tq, tau = theta
    sfr = expsfh(tau, tq, t)
    nuv_u = N.zeros_like(sfr)
    u_r = N.zeros_like(sfr)
    # Work out total flux at each time given the sfh model of tau and tq (calls fluxes function)
    total_flux = fluxes.assign_total_flux(data[0,1:], data[1:,0], data[1:,1:], t*1E9, sfr)
    # Calculate fluxes from the flux at all times then interpolate to get one colour for the age you are observing the galaxy at - if many galaxies are being observed, this also works with an array of ages to give back an array of colours
    nuv_u, u_r = get_colours(t*1E9, total_flux, data)
    nuv_u_age = N.interp(age, t, nuv_u)
    u_r_age = N.interp(age, t, u_r)
    return nuv_u_age, u_r_age

#Calculate colours and magnitudes for functions above
def get_colours(time_steps, sfh, data):
    nuvmag = get_mag(time_steps, sfh, nuvwave, nuvtrans, data)
    umag = get_mag(time_steps, sfh, uwave, utrans, data)
    rmag = get_mag(time_steps, sfh, rwave, rtrans, data)
    nuv_u = nuvmag - umag
    u_r = umag - rmag
    return nuv_u, u_r

def get_mag(time_steps, total_flux, wave, trans, data):
    mag = fluxes.calculate_AB_mag(time_steps, data[1:,0],total_flux, wave, trans)
    return mag

# Function for likelihood of model given all galaxies
def lnlike_one(theta, ur, sigma_ur, nuvu, sigma_nuvu, age):
    tq, tau = theta
    pred_nuvu, pred_ur = predict_c_one(theta, age)
    #inv_sigma_ur = 1./((sigma_ur**2)*(2*N.pi))**0.5
    #inv_sigma_nuvu = 1./((sigma_nuvu**2)*(2*N.pi))**0.5
    #return N.log(inv_sigma)-0.5*((ur-pred_ur)**2/sigma_ur**2) #- 0.5*((nuvu-pred_nuvu)**2/sigma_nuvu**2)
    return -0.5*((ur-pred_ur)**2/sigma_ur**2)-0.5*((nuvu-pred_nuvu)**2/sigma_nuvu**2)

# Function which includes GZ likelihoods and sums across all galaxies to return one value for a given set of theta 
def lnlike(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    ts, taus, td, taud = theta
    d = lnlike_one([td, taud], ur, sigma_ur, nuvu, sigma_nuvu, age)
    s = lnlike_one([ts, taus], ur, sigma_ur, nuvu, sigma_nuvu, age)
    D = N.log(pd) + d
    S = N.log(ps) + s
    return N.sum(N.logaddexp(D, S))

# Prior likelihood on theta values given the inital w values assumed for the mean and stdev
def lnprior(w, theta):
    mu_tqs, mu_taus, mu_tqd, mu_taud, sig_tqs, sig_taus, sig_tqd, sig_taud = w
    ts, taus, td, taud = theta
    if 0.0 < ts < 13.807108309208775 and 0.0 < taus < 5.0 and 0.0 < td < 13.807108309208775 and 0.0 < taud < 5.0:
        ln_tqs = - 0.5*((ts-mu_tqs)**2/sig_tqs**2) #- N.log((2*N.pi*sig_tqs**2)**0.5)
        ln_taus = - 0.5*((taus-mu_taus)**2/sig_taus**2) #-N.log((2*N.pi*sig_taus**2)**0.5)
        ln_tqd = - 0.5*((td-mu_tqd)**2/sig_tqd**2) #-N.log((2*N.pi*sig_tqd**2)**0.5) 
        ln_taud = - 0.5*((taud-mu_taud)**2/sig_taud**2) #-N.log((2*N.pi*sig_taud**2)**0.5)
        #print 'prior', ln_tqs + ln_taus + ln_tqd + ln_taud
        return ln_tqs + ln_taus + ln_tqd + ln_taud
    else:
        return -N.inf

# Overall likelihood function combining prior and model
def lnprob(theta, w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    global n
    n+=1
    if n %100 == 0:
        print 'step number', n/100
    lp = lnprior(w, theta)
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike(theta, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps)


def sample(ndim, nwalkers, nsteps, burnin, start, w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps):
    if len(age) != len(ur):
        raise SystemExit('Number of ages does not coincide with number of galaxies...')
    p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=(w, ur, sigma_ur, nuvu, sigma_nuvu, age, pd, ps))
    #burn in 
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau/gv/samples_gv_burn_in_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
    N.save(samples_save, samples)
    walker_plot(samples, nwalkers, burnin)
    sampler.reset()
    print 'RESET', pos
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    samples_save = '/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau/gv/samples_gv_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
    N.save(samples_save, samples)
    fig = triangle.corner(samples, labels=[r'$ t_{smooth} $', r'$ \tau_{smooth} $', r'$ t_{disc} $', r'$ \tau_{disc}$'])
    fig.savefig('triangle_t_tau_gv_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')
    return samples, fig, samples_save


#Define function to plot the walker positions as a function of the step
def walker_plot(samples, nwalkers, limit):
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
#    ax1.set_ylim(0, 13.807108309208775)
#    ax2.set_ylim(0, 3.0)
#    ax3.set_ylim(0, 13.807108309208775)
#    ax4.set_ylim(0, 3.0)
    ax4.set_xlabel(r'step number')
    ax1.set_ylabel(r'$t_{smooth}$')
    ax2.set_ylabel(r'$\tau_{smooth}$')
    ax3.set_ylabel(r'$t_{disc}$')
    ax4.set_ylabel(r'$\tau_{disc}$')
    P.subplots_adjust(hspace=0.1)
    save_fig = '/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau/gv/walkers_steps_gv_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
    fig.savefig(save_fig)
    return fig

def walker_steps(samples, nwalkers, limit):
    ur = N.load('/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau_old/colour_plot/ur.npy')
    nuv = N.load('/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau_old/colour_plot/nuvu.npy')
    s = samples.reshape(nwalkers, -1, 4)
    s = s[:,:limit,:]
    fig = P.figure(figsize=(9,9))
    ax1 = P.subplot(221, autoscale_on = False, aspect='auto', xlim=[0,13.8], ylim=[0,3])
    ax2 = P.subplot(222,  autoscale_on = False, aspect='auto', xlim=[0,13.8], ylim=[0,3])
    ax3 = P.subplot(223,  autoscale_on = False, aspect='auto', xlim=[0,13.8], ylim=[0,3])
    ax4 = P.subplot(224,  autoscale_on = False, aspect='auto', xlim=[0,13.8], ylim=[0,3])
    ax1.imshow(ur, origin='lower', aspect='auto', extent=[0, 13.8, 0, 3])
    ax2.imshow(ur, origin='lower', aspect='auto', extent=[0, 13.8, 0, 3])
    ax3.imshow(nuv, origin='lower', aspect='auto', extent=[0, 13.8, 0, 3])
    ax4.imshow(nuv, origin='lower', aspect='auto', extent=[0, 13.8, 0, 3])
    for n in range(len(s)):
        ax1.plot(s[n,:,0], s[n,:,1], 'k', alpha=0.5)
        ax2.plot(s[n,:,2], s[n,:,3], 'k', alpha=0.5)
        ax3.plot(s[n,:,0], s[n,:,1], 'k', alpha=0.5)
        ax4.plot(s[n,:,2], s[n,:,3], 'k', alpha=0.5)
    ax1.set_xlabel(r'$t_{smooth}$')
    ax1.set_ylabel(r'$\tau_{smooth}$')
    ax2.set_xlabel(r'$t_{disc}$')
    ax2.set_ylabel(r'$\tau_{disc}$')
    ax3.set_xlabel(r'$t_{smooth}$')
    ax3.set_ylabel(r'$\tau_{smooth}$')
    ax4.set_xlabel(r'$t_{disc}$')
    ax4.set_ylabel(r'$\tau_{disc}$')
    P.tight_layout()
    save_fig = '/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau/gv/walkers_2d_steps_gv_'+str(nwalkers)+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
    fig.savefig(save_fig)
    return fig

def plot_binned_data(axes, binedges, data, *args, **kwargs):
    #The dataset values are the bin centres
    x = (binedges[1:] + binedges[:-1]) / 2.0
    #The weights are the y-values of the input binned data
    weights = data
    return axes.hist(x, bins=binedges, weights=weights, *args, **kwargs)

def corner_plot(s, labels, extents, bf):
    x, y = s[:,0], s[:,1]
    fig = P.figure(figsize=(10,10))
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
    #n= ax1.hist(x, bins=100, range=(extents[0][0], extents[0][1]), normed=True, histtype='step', color='k')
    den = kde.gaussian_kde(x[N.logical_and(x>=extents[0][0], x<=extents[0][1])])
    pos = N.linspace(extents[0][0], extents[0][1], 750)
    ax1.plot(pos, den(pos), 'k-', linewidth=1)
    ax1.axvline(x=bf[0][0], linewidth=1)
    ax1.axvline(x=bf[0][0]-bf[0][1], c='b', linestyle='--')
    ax1.axvline(x=bf[0][0]+bf[0][2], c='b', linestyle='--')
    #   ax1.set_ylim(0, 1)
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
    ax3.axhline(y=bf[1][0]-bf[1][1], c='b', linestyle='--')
    ax3.axhline(y=bf[1][0]+bf[1][2], c='b', linestyle='--')
    #    ax3.set_xlim(0, 1)
    ax3.set_ylim(extents[1][0], extents[1][1])
    P.subplots_adjust(wspace=0.0)
    P.subplots_adjust(hspace=0.0)
    #    cbar_ax = fig.add_axes([0.67, 0.64, 0.02, 0.26])
    #    cb = fig.colorbar(im, cax = cbar_ax, ticks=[0.015, 0.045, 0.075, 0.105, 0.135])
    #    cb.solids.set_edgecolor('face')
    #    cb.set_label(r'predicted SFR $[M_{\odot} yr^{-1}]$', labelpad = 20, fontsize=16)
    #P.tight_layout()
    return fig




#Load the filters in order to calculate fluxes in each bandpass
filters = idlsave.read('/Volumes/Data/smethurst/Green-Valley-Project/kevin_idl/ugriz.sav')
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
