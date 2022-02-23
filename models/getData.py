import numpy as np
from scipy.special import erf
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

def redshiftGrid(zMax,nPoints):

    c = 299792458 # m/s
    H_0 = 67900.0 # m/s/MPc
    Omega_M = 0.3065 # unitless
    Omega_Lambda = 1.0-Omega_M
    year = 365.25*24.*3600

    cosmo = Planck15.clone(name='cosmo',Om0=Omega_M,H0=H_0/1e3,Tcmb0=0.)
    zs = np.linspace(0,zMax,nPoints)
    dVc_dz = 4.*np.pi*cosmo.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value

    return zs,dVc_dz

def gaussian(samples,mu,sigma,lowCutoff,highCutoff):
    a = (lowCutoff-mu)/np.sqrt(2*sigma**2)
    b = (highCutoff-mu)/np.sqrt(2*sigma**2)
    norm = np.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    return np.exp(-(samples-mu)**2/(2.*sigma**2))/norm

def getInjections(mu_chi=0.05,
    sig_chi=0.1):

    injectionFile = "/Users/tcallister/Documents/Repositories/o3b-pop-studies/code/injectionDict_10-20_directMixture_FAR_1_in_1.pickle"
    injectionDict = np.load(injectionFile,allow_pickle=True)

    Xeff_det = np.array(injectionDict['Xeff'])
    inverse_inj_weights = np.array(injectionDict['weights_XeffOnly'])

    pop_reweight = inverse_inj_weights
    injectionDict['pop_reweight'] = pop_reweight

    return injectionDict

def getSamples(mu_chi=0.05,
    sig_chi=0.1):

    # Dicts with samples:
    sampleDict = np.load("/Users/tcallister/Documents/LIGO-Data/sampleDict_FAR_1_in_1_yr.pickle",allow_pickle=True)
    sampleDict.pop('S190814bv')

    for event in sampleDict:
        print(event,len(sampleDict[event]['m1']))

    nEvents = len(sampleDict)
    for event in sampleDict:

        Xeff = np.array(sampleDict[event]['Xeff'])
        z = np.array(sampleDict[event]['z'])
        p_Xeff = gaussian(Xeff,mu_chi,sig_chi,-1,1)

        c = 299792458 # m/s
        H_0 = 67900.0 # m/s/MPc
        Omega_M = 0.3065 # unitless

        cosmo = Planck15.clone(name='cosmo',Om0=Omega_M,H0=H_0/1e3,Tcmb0=0.)
        dVc_dz = 4.*np.pi*cosmo.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value
        sampleDict[event]['dVdz'] = dVc_dz
        
        sampleDict[event]['weights_over_priors'] = sampleDict[event]['weights']/sampleDict[event]['Xeff_priors']
        ps = np.ones(Xeff.size)
        ps /= np.sum(ps)

        inds_to_keep = np.random.choice(np.arange(Xeff.size),size=2000,replace=True,p=ps)
        for key in sampleDict[event].keys():
            sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    return sampleDict

if __name__=="__main__":
    getSamples()
