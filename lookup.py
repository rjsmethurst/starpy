from posterior import *
from itertools import product
import numpy as N

""" Function to generate a look up table of u-r and NUV-u colours using the predict_c_one funciton in StarfPy. Defaults are for a 50 x 100 x 100 look up table in age, tau and t. 
    """

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
        print '% complete: ', (float(n)/len(grid))*100
    nuv[n], ur[n] = predict_c_one([grid[n,2], grid[n,1]], grid[n,0])
    N.save('nuv_look_up.npy', nuv)
    N.save('ur_look_up.npy', ur)
