"""This code loads in the *.rhb files from the M05 models and extracts the 67 ages, 1221 wavelength values and the corresponding flux values at each wavelength for each of the 67 ages. It has been defined as a function to be called in order to generate the extracted .rhb file for a specified model file that is input in the function."""

#############################################################################
#
# INPUTS
# M05 model *.rhb files - both low resolution with 1221 wavelengths or high resolution with 6900 wavelengths
#
# OUTPUTS
# ages array - with dimensions (221,1)
# wavelengths array - with dimensions (1221, 1) or (6900, 1) depending on lr/hr
# SEDs array - with dimensions (1222, 222) or (6901, 222) depending on lr/hr - the first row contains the ages which correspond to the column below it (the first entry in this row is 0 - the first column (seds[:,0]) contains the wavelength values and all the following columns contain the flux values for the corresponding wavelength at each age value in the ages_array. The array therefore looks something like this:
#                       0     a1      a2      a3      ...
#                       W1    f1,1    f2,1    f3,1    ...
#                       W2    f1,2    f2,2    f3,2   ...
#                       W3    f1,3    f2,3    f3,3   ...
#                       ...   ...     ...     ...     ...
#
#############################################################################

import numpy as N
import os 

def extract_model_data(modelfile):
#Open the file loaded into the funciton and read each line
    if os.path.exists(modelfile) == False:
        pass
    else:
        p = N.loadtxt(modelfile)
        num_wave = 1221
        num_ages = N.unique(p[:,0]).shape[0]
        extract = N.zeros((num_wave+1, num_ages+1,))
        extract[1:,0] = p[0:num_wave, 2]
        extract[0,1:] = N.unique(p[:,0])*1E9 # convert to Gyr
        n=0
        for j in range(num_ages):
            extract[1:,j+1] = p[n:n+num_wave, -1]
            n+=num_wave
#This code saves the sed
        if 'kr' in modelfile:
            new_dir = dir+'/Sed_Mar05_SSP_Kroupa/extracted/'
        else:
                new_dir = dir+'/Sed_Mar05_SSP_Salpeter/extracted/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        else:
            pass
        N.savetxt(new_dir+'extracted_'+modelfile[-14:], extract, fmt='%1.8E')

### Extract all the M05 files: all combinations of chabrier & salpeter IMFs, high res & low res and all metallicities
metal = ['z007', 'z004', 'z002', 'z001', 'z0001', 'z10m4']
dir = str(raw_input('Location of downloaded SPS files e.g. "~/m05/" : '))
for n in range(len(metal)):
    extract_model_data(dir+'/Sed_Mar05_SSP_Kroupa/sed.kr'+metal[n]+'.rhb')
    extract_model_data(dir+'/Sed_Mar05_SSP_Kroupa/sed.kr'+metal[n]+'.bhb')
    extract_model_data(dir+'/Sed_Mar05_SSP_Salpeter/sed.ss'+metal[n]+'.bhb')
    extract_model_data(dir+'/Sed_Mar05_SSP_Salpeter/sed.ss'+metal[n]+'.rhb')
    
