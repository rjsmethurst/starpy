"""This code loads in the *.ised_ASCII files from the BC03 models and extracts the 221 ages, 1221/6900 wavelength values and the corresponding flux values at each wavelength for each of the 221 ages. It has been defined as a function to be called in order to generate the extracted ASCII file for a specified model file that is input in the function."""

#############################################################################
#
# INPUTS
# BC03 model *ised_ASCII files - both low resolution with 1221 wavelengths or high resolution with 6900 wavelengths
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
import pylab as P
import os 

def extract_model_data(modelfile):
    #modelfile='bc2003_lr_m62_chab_ssp.ised_ASCII'
    if 'chab' in modelfile:
        dir = '/Users/becky/Projects/Green-Valley-Project/bc03/models/Padova1994/chabrier/'
    else:
        dir = '/Users/becky/Projects/Green-Valley-Project/bc03/models/Padova1994/salpeter/'
#Open the file loaded into the funciton and read each line
    file = open(dir+modelfile, 'r')
    lines = file.readlines()
    file.close()
#Determine whether the file is high resolution or low resolution and therefore define the number of wavelengths which will be present
    if 'lr' in modelfile:
        num_wave = 1221
    else:
        num_wave = 6900
#The number of ages is constant across all the model files
    num_ages = 221
#Remove the 5 lines in the middle of each file which contain text defining the model parameters - these prevent the code from floating the data and making the correct arrays
    if 'chab' in modelfile:
        lines_step = lines[0:37]
        new_lines = N.append(lines_step, lines[42:])
    if 'salp' in modelfile:
        lines_step=lines[0:37]
        new_lines = N.append(lines_step, lines[43:])

#Split the lines so that the data contains one large array of all the data in a row
    initial_data=[]
    for n in range(len(new_lines)):
        p = new_lines[n].split()
        initial_data = N.append(initial_data,p)
#Float the data and reshape the array into a single column of data.
    float_data = initial_data.astype(float)
    data = float_data.reshape(len(float_data),1)
#Define the function which returns the correct section of the data and a new data array with that section of data removed. The data contains some odd numbers after the fluxes have been given, each starting with a 52 and containing only 50 or so entries - a column of zeros is returned for these sections and they are deleted from the data array (this occurs under the else command)
    def create_arrays(data_array):
        if data_array[0] > num_ages-1:
            new_array = data_array[1:int(data_array[0])+1]
            n_data =[]
            n_data = N.append(n_data, data_array[len(new_array)+1:])
            new_data = n_data.reshape(len(n_data),1)
        else:
            n_array= N.zeros(num_wave)
            new_array = n_array.reshape(len(n_array), 1)
            n_data=[]
            n_data= N.append(n_data, data_array[int(data_array[0])+1:])
            new_data = n_data.reshape(len(n_data),1)
        return new_array, new_data
#Extract the ages and wavelenth arrays using the above function
    age, working_data = create_arrays(data)
    wavelength, work_data = create_arrays(working_data)
# Begin the definition of the sed array by setting the first column equal to the wavelength array then append this array with every new flux array that is created by implementing the above function
    sed=wavelength
    for n in range(len(age)*2):
        flux, work_data = create_arrays(work_data)
        sed = N.append(sed, flux,1)
#The following code removes the columns of zeros from the sed array by creating a new aray seds which does not contain these columns. It's final dimensions are (1221/6900 rows by 222 columns) with the first column defining the wavelengths and the rest of the columns the flux at that wavelength at each of the 221 age values defined in the ages array
    c=N.zeros_like(sed)
    n=0
    for i in range(sed.shape[1]):
        if sed[:,i].any()==True:
            c[:,n] = N.copy(sed[:,i])
            n+=1
        else:
            pass
    seds=c[:,0:n]
#We now must append the seds array so that the top row reads the age of the fluxes which follow in the relevant column.
    ages = N.zeros(len(age)+1)
    ages=ages.reshape(len(ages),1)
    for j in range(len(age)):
        ages[j+1,0]=age[j]
    ages=ages.reshape(1, len(ages))
    seds = N.insert(seds, 0, ages, axis=0)

    print len(seds)
    print N.shape(seds)


#This code saves the sed
    if 'chab' in modelfile:
        new_dir = dir+'/Padova1994/chabrier/ASCII/'
    else:
            new_dir = dir+'/Padova1994/salpeter/ASCII/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        pass
    N.savetxt(dir+'extracted_'+modelfile,seds,fmt='%1.8E')

### Extract all the BC03 files: all combinations of chabrier & salpeter IMFs, high res & low res and all metallicities
metal = [22,32,42,52,62,72]
for n in range(len(metal)):
    dir = str(raw_input('Location of downloaded SPS files e.g. "~/bc03/models" : '))
    extract_model_data(dir+'/Padova1994/salpeter/bc2003_hr_m'+str(metal[n])+'_salp_ssp.ised_ASCII')
    extract_model_data(dir+'/Padova1994/salpeter/bc2003_lr_m'+str(metal[n])+'_salp_ssp.ised_ASCII')
    extract_model_data(dir+'/Padova1994/chabrier/bc2003_hr_m'+str(metal[n])+'_chab_ssp.ised_ASCII')
    extract_model_data(dir+'/Padova1994/chabrier/bc2003_lr_m'+str(metal[n])+'_chab_ssp.ised_ASCII')
    
