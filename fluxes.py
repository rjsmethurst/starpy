import numpy as N
import scipy as S
from scipy.integrate import simps
from scipy import interpolate
import pylab as P

dir ='/Volumes/Data/smethurst/Green-Valley-Project/bc03/models/Padova1994/chabrier/ASCII/'
model = 'extracted_bc2003_lr_m62_chab_ssp.ised_ASCII'
data = N.loadtxt(dir+model)
model_ages = data[0,1:]
model_lambda = data[1:,0]
model_fluxes = data[1:,1:]
time = N.arange(0, 0.01, 0.003)
t = N.linspace(0,14.0,100)
time_steps = N.append(time, t[1:])*1E9
#First mask the ages of the very young stars hidden in birth clouds
mask = model_ages[model_ages<4E6]
model_fluxes[:,0:len(mask)] = 0.0
# Calculate the fluxes at the ages specified by the time steps rather than in the models using numpy/scipy array manipulations rather than a for loop
f = interpolate.interp2d(model_ages, model_lambda, model_fluxes)
interp_fluxes_sim = f(time_steps, model_lambda)

def assign_total_flux(model_ages, model_lambda, model_fluxes, time_steps, sim_SFR):
#    ##First mask the ages of the very young stars hidden in birth clouds
#    mask = model_ages[model_ages<4E6]
#    model_fluxes[:,0:len(mask)] = 0.0
#    ## Calculate the fluxes at the ages specified by the time steps rather than in the models using numpy/scipy array manipulations rather than a for loop
#    f = interpolate.interp2d(model_ages, model_lambda, model_fluxes)
#    interp_fluxes_sim = f(time_steps, model_lambda)
    # Produce the array to keep track of the ages of the fractional SFR at each time step
    frac_sfr = sim_SFR/sim_SFR[0]
    fraction_array = S.linalg.toeplitz(frac_sfr, N.zeros_like(frac_sfr)).T
    # Produce the array to keep track of the ages of the mass fraction of stars formed at each time step
    m_array = (sim_SFR.T)*(N.append(1, N.diff(time_steps)))
    mass_array = S.linalg.toeplitz(m_array, N.zeros_like(frac_sfr)).T
    # Produce the array to keep track of the fraction of flux produced at each timestep
    frac_flux_array = fraction_array*mass_array
    # Calculate the total flux contributed by all of the fractions at each time step by summing across all wavelength values
    flux_lambda = frac_flux_array*(N.split(interp_fluxes_sim.T, len(model_lambda), axis=1))
    total_flux = (N.sum(flux_lambda, axis=1)).T # Array of dimensions (len(timesteps), len(model_lambda))
    return total_flux


def calculate_AB_mag(time_steps, model_lambda, sim_flux, wave, trans):
    lambda_filter1 = [i for i in model_lambda if i > wave[0] and i < wave[len(wave)-1]]
    lambda_filter2 = N.append(wave[0], lambda_filter1)
    lambda_filter = N.append(lambda_filter2, wave[len(wave)-1])
    
    f = interpolate.interp2d(model_lambda, time_steps, sim_flux)
    flux_filter = (f(lambda_filter, time_steps))
    trans_filter = N.interp(lambda_filter, wave, trans)
    
    top = N.trapz((lambda_filter*flux_filter*trans_filter), lambda_filter, axis=1)
    bottom = N.trapz(trans_filter/lambda_filter, lambda_filter)
    
    m_ab = -2.41 - 2.5*N.log10(top/bottom)
    return m_ab

