from posterior import *
from astropy.cosmology import FlatLambdaCDM
from itertools import product

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

tq = N.linspace(0.003, 13.8, 100)
tau = N.linspace(0.003, 4, 100)
age = N.linspace(10.88861228, 13.67023409, 50)

print 'making product list of inputs...'
grid = N.array(list(product(age, tau, tq)))

ur = N.zeros(len(grid))
nuv = N.zeros(len(grid))

'calculating colours...'
for n in range(len(grid)):
    if n%10000 == 0:
        print '% complete: ', float(n)/len(grid)
    nuv[n], ur[n] = predict_c_one([grid[n,2], grid[n,1]], grid[n,0])

'saving...'
N.save('nuv_look_up.npy', nuv)
N.save('ur_look_up.npy', ur)

    
