from posterior import *
from astropy.cosmology import FlatLambdaCDM

import numpy as N
import pylab as PY
import pyfits as F
import os
import time


font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

#reason = str(raw_input('Why are you running this iteration? : '))
#
##Using PyFits to open up the Galaxy Zoo data
#file = '/Volumes/Data/smethurst/Green-Valley-Project/data/GZ2_all_GALEX_match_GZ1_k_correct_green_valley.fits'
#dat = F.open(file)
#gz2data = dat[1].data
#dat.close()
#
#col = N.zeros(12*len(gz2data)).reshape(len(gz2data), 12)
#col[:,0] = gz2data.field('MU_MR')
#col[:,1] = gz2data.field('NUV_U')
#col[:,2] = gz2data.field('t01_smooth_or_features_a01_smooth_debiased')
#col[:,3] = gz2data.field('t01_smooth_or_features_a02_features_or_disk_debiased')
#col[:,4] = ((gz2data.field('Err_MU_MR'))**2 + 0.05**2)**0.5
#col[:,5] = ((gz2data.field('Err_NUV_U'))**2 + 0.1**2)**0.5
#col[:,6] = gz2data.field('z_1')
#col[:,7] = gz2data.field('zErr_1')
#col[:,8] = gz2data.field('GV_first')
#col[:,9] = gz2data.field('GV_sec')
#col[:,10] = gz2data.field('upper_GV')
#col[:,11] = gz2data.field('lower_GV')
#
#
#non_nan = N.logical_not(N.isnan(col[:,1])).astype(int)
#colours = N.compress(non_nan, col, axis=0)
#gvf = colours[colours[:,8]==1]
#colours = colours[colours[:,0] < 6.0]
#colours = colours[colours[:,1] > -3.0]
#gv = gvf[gvf[:,9]==1]
#print len(gv)
#smooth = colours[colours[:,2] > 0.8]
#print len(smooth)
#disc = colours[colours[:,3] > 0.8]
#print len(disc)
#red_s = colours[colours[:,0] > colours[:,10]]
#print len(red_s)
#blue_c = colours[colours[:,0] < colours[:,11]]
#print len(blue_c)
#
age_save = '/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau_old/gv/age_gv.npy'
age_path = os.path.exists(age_save)
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)
if age_path ==False:
    age = N.array(cosmo.age(gv[:,6]))
    N.save(age_save, age)
else:
    age = N.load(age_save)
print len(age)

w = [7.5, 1.5, 7.0, 1.5, 4.0, 1.5, 4.0, 1.5]
nwalkers = 100
nsteps= 1000
start = [7.5, 1.5, 7.5, 1.5]
burnin =1000

#f = open('/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau_old/log.txt', 'a')
#f.write('Run started at '+str(time.strftime('%H:%M'))+' on '+str(time.strftime('%d/%m/%y'))+'\n')
#f.write('Reason for running iteration: ' +reason+'\n')
#f.write('Number of walkers : '+str(nwalkers)+'\n')
#f.write('Number of steps :'+str(nsteps)+'\n')
#f.write('Starting point of walkers : '+str(start)+'\n')
#f.write('Prior assumptions, w :'+str(w)+'\n')
#f.write('Number of galaxies used was : '+str(len(age))+'\n')
#f.write('Ages of galaxies used found at : '+str(age_save)+'\n')
#f.close()
#
## w is the prior conditions on my theta parameters - the mean and standard deviation of tq and tau for the disc and smooth populations
## w = [mu_tqs, mu_taus, mu_tqd, mu_taud, sig_tqs, sig_taus, sig_tqd, sig_taud]
#start_time = time.time()
##The rest calls the emcee code and makes plots....
#samples, fig, samples_save = sample(4, nwalkers, nsteps, burnin, start, w, gv[:,0], gv[:,4], gv[:, 1], gv[:, 5], age, gv[:,3], gv[:,2])
#elap = (time.time() - start_time)/60
#print 'Minutes taken for '+str(len(samples)/nwalkers
#                               )+' steps and '+str(nwalkers)+' walkers', elap

samples =N.load('samples_gv_100000_23944_06_55_17_06_14.npy')

sl_s = samples[:,0:2][N.logical_and(samples[:,0] >=8.6, samples[:,0]<= 8.66)]
sl_d = samples[:,2:4][N.logical_and(samples[:,2]>=8.465, samples[:,2] <=8.495)]

tqs_mcmc, taus_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(sl_s, [16,50,84],axis=0)))
tqd_mcmc, taud_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(sl_d, [16,50,84],axis=0)))

print 'tq_smooth',tqs_mcmc
print 'tau_smooth',taus_mcmc
print 'tq_disc',tqd_mcmc
print 'tau_disc',taud_mcmc

#f = open('/Volumes/Data/smethurst/Green-Valley-Project/bayesian/find_t_tau_old/log.txt', 'a')
#f.write('Samples found at : '+str(samples_save)+'\n')
#f.write('Total number of positions for each parameter : '+str(len(samples))+'\n')
#f.write('t_smooth from MCMC : '+str(tqs_mcmc)+'\n')
#f.write('tau_smooth from MCMC : '+str(taus_mcmc)+'\n')
#f.write('t_disc from MCMC : '+str(tqd_mcmc)+'\n')
#f.write('tau_disc from MCMC : '+str(taud_mcmc)+'\n')
#f.write('Time taken to complete : '+str(elap/60)+' hours \n')
#f.write(' \n')
#f.write('------------------------------------------------------------------ \n')
#f.write(' \n')
#f.close()


#fig = triangle.corner(samples, labels = [r'$ t_{smooth}$', r'$ \tau_{smooth}$', r'$t_{disc}$', r'$\tau_{disc}$'])
#fig.savefig('triangle_t_tau_gv_'+str(len(samples))+'_'+str(len(age))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')

fig_s = corner_plot(samples[:,0:2], labels = [r'$ t_{smooth}$', r'$ \tau_{smooth}$'], extents=[[8.6, 8.66],[1.52,1.545]], bf=[tqs_mcmc, taus_mcmc])
fig_s.savefig('triangle_t_tau_smooth_gv_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')
fig_d = corner_plot(samples[:,2:4], labels = [r'$t_{disc}$', r'$\tau_{disc}$'], extents=[[8.465, 8.495],[1.168, 1.178]], bf=[tqd_mcmc, taud_mcmc])
fig_d.savefig('triangle_t_tau_disc_gv_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')
#
fig_samp = walker_plot(samples, nwalkers, nsteps)
fig_2 = walker_steps(samples, nwalkers, nsteps)

