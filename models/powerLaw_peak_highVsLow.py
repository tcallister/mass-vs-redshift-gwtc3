import numpyro
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import erf
from jax import vmap

def model(sampleDict,injectionDict,z_grid,dVdz_grid):
    
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa_high = numpyro.sample("kappa_high",dist.Normal(0,3))
    kappa_low = numpyro.sample("kappa_low",dist.Normal(0,3))
    mMin = 5.
    mMax = 100.
    alpha = numpyro.sample("alpha",dist.Normal(-2,4))
    mu_m = numpyro.sample("mu_m",dist.Uniform(25,45))
    sig_m = numpyro.sample("sig_m",dist.Uniform(1,10))
    f_peak = numpyro.sample("f_peak",dist.Uniform(0,1))

    # Compute normalization
    norm_high = jnp.trapz(dVdz_grid*(1.+z_grid)**(kappa_high-1),z_grid)
    norm_low = jnp.trapz(dVdz_grid*(1.+z_grid)**(kappa_low-1),z_grid)
    norm_prior = jnp.trapz(dVdz_grid*(1.+z_grid)**(2.7-1),z_grid)

    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    pop_reweight = injectionDict['pop_reweight']

    p_m1_det_pl = (1.+alpha)*m1_det**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_det_peak = jnp.exp(-(m1_det-mu_m)**2./(2.*np.pi*sig_m**2))/jnp.sqrt(2.*np.pi*sig_m**2.)
    p_m1_det = f_peak*p_m1_det_peak + (1.-f_peak)*p_m1_det_pl
    p_m1_det = jnp.where(m1_det>mMax,0.,p_m1_det)

    p_z_det = dVdz_det*(1.+z_det)**(kappa-1)/norm
    
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq) - mMin**(1.+bq))
    p_m2_det = jnp.where(m2_det<mMin,0.,p_m2_det)
    xi_weights = p_m1_det*p_m2_det*p_z_det*pop_reweight
    
    nEff_inj = jnp.sum(xi_weights)**2/jnp.sum(xi_weights**2)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/69.)

    xi = jnp.sum(xi_weights)
    numpyro.factor("xi",-69*jnp.log(xi))
    
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,weights):

        p_m1_pl = (1.+alpha)*m1_sample**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
        p_m1_peak = jnp.exp(-(m1_sample-mu_m)**2./(2.*np.pi*sig_m**2))/jnp.sqrt(2.*np.pi*sig_m**2.)
        p_m1 = f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl
        p_m1 = jnp.where(m1_sample>mMax,0.,p_m1)
        
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq) - mMin**(1.+bq))
        p_m2 = jnp.where(m2_sample<mMin,0.,p_m2)

        p_z = (1.+z_sample)**(kappa)/norm
        #p_z_prior = (1.+z_sample)**(2.7)/norm_prior

        mc_weights = p_m1*p_m2*p_z*weights#/p_z_prior
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVdz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['weights_over_priors'] for k in sampleDict]))
        
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))
    numpyro.factor("logp",jnp.sum(log_ps))
