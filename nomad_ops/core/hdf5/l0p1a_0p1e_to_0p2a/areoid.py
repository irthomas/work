"""
    Functions to compute the Mars areoid

"""


import os
import logging

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import PFM_AUXILIARY_FILES    
    
__project__   = "NOMAD"
__author__    = "Justin Erwin"
__contact__   = "justin.erwin@aeronomie.be"

logger = logging.getLogger( __name__ )

_mgm1025_filename = os.path.join(PFM_AUXILIARY_FILES, 'areoid_model', 'mgm1025')
RPLANET = 3396.e3

class Model(object):
    """ container for areoid model (ie mgm1025) """

    def __init__(self, filename=None):
        
        import numpy as np

        self.ae=3396.e3
        self.gm=42828.36e9
        self.omega=0.70882187e-4
        self.v0 = 12652777.9
        self.Nmax = 0
        self.Mmax = 0
        self.coefc = np.zeros((self.Mmax+1,self.Nmax+1))
        self.coefs = np.zeros((self.Mmax+1,self.Nmax+1))

        #
        if filename is not None:
            self.read_coeffs(filename)
            #print('success')

    def read_coeffs(self, filename):

        import numpy as np
        from scipy.special import lpmn

        #logger.info("Reading in areoid model: %s"%filename)
        self.Nmax = 50
        self.Mmax = 50

        with open(filename, 'r') as f:
            lines = f.readlines()

        #logger.info("Description: %s"%lines[0][:-1])

        #
        _,_,Nmax, Mmax, temp = lines[1].split()
        self.gm = np.float(temp[:20])
        self.ae = np.float(temp[20:35])
        self.Nmax = int(Nmax)
        self.Mmax = int(Mmax)
        
        self.coefc = np.zeros((self.Mmax+1,self.Nmax+1))
        self.coefs = np.zeros((self.Mmax+1,self.Nmax+1))
        
        #
        for line in lines[2:]:
            gcoef, n, m, coef = line.split()
            n = int(n)
            m = int(m)
            coef = float(coef)
            if gcoef == 'GCOEFC1':
                self.coefc[m,n] = coef
            elif gcoef == 'GCOEFS1':
                self.coefs[m,n] = coef
            else:
                raise Exception("mgm1025 line error %s"%gcoef)

        #
        r = RPLANET
        xi = self.ae/r
        P0n = lpmn(0, self.Nmax, 0.0)[0]
        K0n = np.array([np.sqrt(2.*n+1) for n in np.arange(0,self.Nmax+1)])
        sum_c = 1. + (self.coefc[0,2:] * xi**np.arange(2,self.Nmax+1) * P0n[0,2:]*K0n[2:]).sum()
        self.v0 = self.gm*sum_c/r + 0.5*self.omega**2*r**2



_mgm1025 = Model(_mgm1025_filename)
            

def geoid(lon, lat, method='interp', **kwargs):

    if method == 'interp':
        return geoid_interp(lon, lat, **kwargs)
    elif method == 'lpmn':
        return geoid_lpmn(lon, lat, **kwargs)
    else:
        raise Exception("method=%s not recognized"%method)
        return -999


def geoid_interp(lon, lat, rescode='4'):
    """ compute mars areoid via interpolation of a precomputed lookup table """

    import os
    import numpy as np
    import h5py

    # choose variable based on rescode (eg resolution)
    if rescode == '4':
        ires = 4.
        areoid_filename = os.path.join(PFM_AUXILIARY_FILES, 'areoid_model', 'mars_areoid_04.h5')
    elif rescode == '16':
        ires = 16.
        areoid_filename = os.path.join(PFM_AUXILIARY_FILES, 'areoid_model', 'mars_areoid_16.h5')
    elif rescode == '32':
        ires = 32.
        areoid_filename = os.path.join(PFM_AUXILIARY_FILES, 'areoid_model', 'mars_areoid_32.h5')
    else:
        raise Exception("rescode=%s not recognized"%rescode)
        return -999

    lon = np.mod(lon, 360.)
    assert lat >= -90. and lat <= 90.
    #print(lon, lat)

    ilon = int(np.floor(lon*ires))
    ilat = int(np.floor((lat+90.)*ires))
    #print(ilon, ilat, lon, lat)

    lon1 = ilon/ires
    lon2 = (ilon+1)/ires
    lat1 = ilat/ires - 90.
    lat2 = (ilat+1)/ires - 90.
    w_lat_lon = bilinear_spherical_interp_weights([lat1,lat2],[lon1,lon2],lat,lon)
    #print(w_lat_lon)

    ra = 3396.e3
    with h5py.File(areoid_filename, 'r') as f:
        ra = np.sum(w_lat_lon*f['ra'][ilat:ilat+2,ilon:ilon+2])

    return ra/1.0e3


def bilinear_spherical_interp_weights(Lats, Lons, lat, lon):
    """ Compute the weights for a Langrange interpolation """
    import numpy as np

    Thetas = np.pi/2. - np.deg2rad(Lats)
    theta = np.pi/2. - np.deg2rad(lat)
    Phis = np.deg2rad(Lons)
    phi = np.deg2rad(lon)

    if Phis[0] > Phis[1]:
        if phi >= Phis[1]:
            phi = phi - 2*np.pi
        Phis[0] = Phis[0] - 2*np.pi
    #print(Thetas, theta, Phis, phi)

    w = np.outer([abs(np.cos(theta)-np.cos(Thetas[1])), abs(np.cos(theta)-np.cos(Thetas[0]))],
                    [abs(phi-Phis[1]), abs(phi-Phis[0])])
    w /= w.sum()

    return w

def geoid_lpmn(lon, lat, model=None):
    """ compute geoid using spherical harmonics"""

    import numpy as np
    from scipy.special import lpmn

    TOL = 0.1

    if model is None:
        model = _mgm1025

    Mmax = model.Mmax
    Nmax = model.Nmax

    #
    rlon = np.deg2rad(lon)
    rlat = np.deg2rad(lat)
    x = np.sin(rlat)
    cslt = np.cos(rlat)

    # compute associated lagrange polynomials
    Pmn = lpmn(Mmax, Nmax, x)[0]

    # "Full/complete normalization II" (Gaposhkin) normalization for "Elementary surface spherical harmonics"
    Kmn = np.zeros((Mmax+1,Nmax+1))
    for n in range(Nmax+1):
        Kmn[0,n] = np.sqrt(2*n+1)
        for m in range(1,n+1):
            #Kmn[m,n] = Kmn[0,n]*np.sqrt(2*np.math.factorial(n-m)/np.math.factorial(n+m))*(-1)**m
            Kmn[m,n] = Kmn[0,n]*np.sqrt(2/np.arange(n-m+1,n+m+1,dtype=float).prod())*(-1)**m

    # loop to compute r
    r = RPLANET
    rg = r
    for i in range(5):
        xi = model.ae/r
        sum_cs = 0.
        for m in range(Mmax+1):
            sum_cs += ((model.coefc[m,m:]*np.cos(m*rlon) + model.coefs[m,m:]*np.sin(m*rlon)) * 
                            Pmn[m,m:]*Kmn[m,m:] * xi**np.arange(m,Nmax+1)).sum()
        sum_cs += 1.

        #
        rg = (model.gm*sum_cs + 0.5*model.omega**2*r**3 * cslt**2)/model.v0
        diff = r - rg

        #
        if abs(diff) < TOL:
            break
        r = rg

    return rg/1.e3


