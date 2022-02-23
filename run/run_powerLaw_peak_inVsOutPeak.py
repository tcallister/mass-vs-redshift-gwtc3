import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import numpy as np
import arviz as az
import sys
sys.path.append("./../models/")
from getData import *
from powerLaw_peak_inVsOutPeak import *

nChains = 3
numpyro.set_host_device_count(nChains)

injectionDict = getInjections()
sampleDict = getSamples()
z_grid,dVdz_grid = redshiftGrid(2.,1000)

kernel = NUTS(model)
mcmc = MCMC(kernel,num_warmup=200,num_samples=1000,num_chains=nChains)

rng_key = random.PRNGKey(2)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,z_grid,dVdz_grid)
mcmc.print_summary()

data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../output/powerLaw_peak_inVsOutPeak.cdf")

