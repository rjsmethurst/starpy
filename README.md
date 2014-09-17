starpy
=======

A python code to determine the quenched star formation history parameters of a galaxy using the MCMC algorithm emcee (http://dan.iel.fm/emcee/current/). Please ensure this module is installed and can be imported into your Python installation.

The sample function implements the emcee EnsembleSampler function for the galaxy colours input. Burn-in is run and calculated for the length specified before the sampler is reset and then run for the length of steps specified. 
        
        :INPUTS:
        :ndim:
        The number of parameters in the model that emcee must find. In this case it always 2 with tq, tau
        
        :nwalkers:
        The number of walkers that step around the parameter space. Must be an even integer number larger than ndim. 
        
        :nsteps:
        The number of steps to take in the final run of the MCMC sampler. Integer.
        
        :burnin:
        The number of steps to take in the inital burn-in run of the MCMC sampler. Integer. 
        
        :start:
        The positions in the tq and tau parameter space to start for both disc and smooth parameters. An array of shape (1,2).
        
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
        
        :RETURNS:
        :samples:
        Array of shape (nsteps*nwalkers, 2) containing the positions of the walkers at all steps for all 4 parameters.
        :samples_save:
        Location at which the :samples: array was saved to. 
        :fig:
        Samples plotted as contours and integrated over to a each one dimensional histogram with median values and 1sigma values either side. The triangle.py module is necessary to produce these plots and is provided in the starfpy            repository. It can also be downloaded here: https://github.com/dfm/triangle.py or with pip install triangle_plot         at the command line.
        
        
        
Data inputs occur with the starpy.py file, all necessary functions are in the posterior.py file. 

You can generate a look up table in two colours with lookup.py or you can use the full functions that take a give SFH and calculate the SED at each time step defined with the fluxes.py file. 

You must also extract the necessary SPS models into ASCII files with the 'extract_bc03.py' file or 'extract_m05.py' file if you wish to use the BC03 or M05 SPS models.  

u-r and NUV-u look up tables have already been generated for 50 observed ages, 100 tau and 100 t values for an exponential decline SFH model for BC03 models. These can be used for a quick starfpy run which is less customised. 
