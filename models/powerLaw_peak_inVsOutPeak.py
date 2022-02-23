import numpyro
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import erf
from jax import vmap

def model(sampleDict,injectionDict,z_grid,dVdz_grid):
    
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa_pl = numpyro.sample("kappa_pl",dist.Normal(0,3))
    kappa_peak = numpyro.sample("kappa_peak",dist.Normal(0,3))
    mMin = 5.
    mMax = 100.
    alpha = numpyro.sample("alpha",dist.Normal(-2,4))
    mu_m = numpyro.sample("mu_m",dist.Uniform(25,45))
    sig_m = numpyro.sample("sig_m",dist.Uniform(1,10))
    logR0_pl = numpyro.sample("log_R0_pl",dist.Uniform(-2,2))
    logR0_peak = numpyro.sample("log_R0_peak",dist.Uniform(-2,2))
    R0_pl = 10.**logR0_pl
    R0_peak = 10.**logR0_peak

    # Compute normalization
    norm_pl = jnp.trapz(dVdz_grid*(1.+z_grid)**(kappa_pl-1),z_grid)
    norm_peak = jnp.trapz(dVdz_grid*(1.+z_grid)**(kappa_peak-1),z_grid)
    norm_prior = jnp.trapz(dVdz_grid*(1.+z_grid)**(2.7-1),z_grid)

    R_tot_pl = R0_pl*norm_pl
    R_tot_peak = R0_peak*norm_peak
    f_pl = R_tot_pl/(R_tot_pl+R_tot_peak)

    nTrials = injectionDict['nTrials']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    pop_reweight = injectionDict['pop_reweight']

    # PL
    p_m1_det_pl = (1.+alpha)*m1_det**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_det_pl = jnp.where(m1_det>mMax,0.,p_m1_det_pl)
    p_z_det_pl = dVdz_det*(1.+z_det)**(kappa_pl-1)/norm_pl

    # Peak
    p_m1_det_peak = jnp.exp(-(m1_det-mu_m)**2./(2.*np.pi*sig_m**2))/jnp.sqrt(2.*np.pi*sig_m**2.)
    p_z_det_peak = dVdz_det*(1.+z_det)**(kappa_peak-1)/norm_peak

    # Combine PL+Peak
    p_m1_z_det = f_pl*p_m1_det_pl*p_z_det_pl + (1-f_pl)*p_m1_det_peak*p_z_det_peak

    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq) - mMin**(1.+bq))
    p_m2_det = jnp.where(m2_det<mMin,0.,p_m2_det)
    xi_weights = p_m1_z_det*p_m2_det*pop_reweight/nTrials
    
    nEff_inj = jnp.sum(xi_weights)**2/jnp.sum(xi_weights**2)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/69.)

    xi = jnp.sum(xi_weights)
    numpyro.factor("xi",-69*jnp.log(xi))

    Tobs = 2.
    Ntot = (R_tot_pl+R_tot_peak)*Tobs
    numpyro.factor("rate",69.*jnp.log(Ntot*xi) - Ntot*xi)

    # Convert log uniform on each R0 to log uniform on total
    numpyro.factor("rateConverstion",jnp.log(R0_pl)+jnp.log(R0_peak)-jnp.log(R0_pl+R0_peak))
    
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,weights):

        # Masses
        p_m1_pl = (1.+alpha)*m1_sample**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
        p_m1_peak = jnp.exp(-(m1_sample-mu_m)**2./(2.*np.pi*sig_m**2))/jnp.sqrt(2.*np.pi*sig_m**2.)
        p_m1_pl = jnp.where(m1_sample>mMax,0.,p_m1_pl)

        # Redshifts
        p_z_pl = (1.+z_sample)**(kappa_pl-1)/norm_pl
        p_z_peak = (1.+z_sample)**(kappa_peak-1)/norm_peak

        # Combine
        p_m1_z = f_pl*p_m1_pl*p_z_pl + (1-f_pl)*p_m1_peak*p_z_peak
        
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq) - mMin**(1.+bq))
        p_m2 = jnp.where(m2_sample<mMin,0.,p_m2)

        p_z_prior = (1.+z_sample)**(2.7-1.)/norm_prior

        mc_weights = p_m1_z*p_m2*weights/p_z_prior
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
